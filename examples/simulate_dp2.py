"""Simulate supernovae on the Rubin DP2 visits and write them in the DP2 DIA catalog schema.

    python examples/simulate_dp2.py [data/dp2_sim_sne] [--n-per-class 1000] [--seed 1024]

Forward-models SN Ia (SALT3) and SN II / Ib/c (Vincenzi et al. 2019 templates) with
lightcurvelynx through the DP2 visit-detector table: every CCD visit covering a source gives
one forced-photometry point, with noise from that visit. Positions are drawn over the area
covered by both DP2 visits and the DP2 DIA catalog, with SFD Milky Way extinction. Objects
without at least ``min_detections`` detections are dropped (a DIA object needs detections).

The output HATS catalog has the DP2 DIA schema, so it reads like real data with
``hyrax_parsnip.RUBIN_DIA_SETTINGS``:

- per object: ``diaObjectId``, ``ra``, ``dec``, ``nDiaSources``
- nested ``diaObjectForcedSource``: ``midpointMjdTai``, ``band``, ``psfDiffFlux``,
  ``psfDiffFluxErr`` (nJy), ``visit`` and the flag columns
- truth: ``type``, ``redshift``, ``t0``, ``mwebv``, ``template``

Needs lightcurvelynx, sncosmo, dustmaps and mocpy (all in desc-td-env at NERSC). Follows
``dp2_sims_visitdetector.ipynb`` in lightcurvelynx_rubin_dp2.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

# lightcurvelynx resolves its download directory (passbands, ...) at import time.
os.environ.setdefault("LIGHTCURVELYNX_DATA_DIR", str(Path(__file__).resolve().parent / "data" / "lightcurvelynx"))
Path(os.environ["LIGHTCURVELYNX_DATA_DIR"]).mkdir(parents=True, exist_ok=True)

import lsdb
import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
import sncosmo
from dustmaps.sfd import SFDQuery
from lightcurvelynx.astro_utils.dustmap import DustmapWrapper
from lightcurvelynx.astro_utils.passbands import PassbandGroup
from lightcurvelynx.astro_utils.snia_utils import (
    DistModFromRedshift,
    X0FromDistMod,
    num_snia_per_redshift_bin,
    snia_volumetric_rates,
)
from lightcurvelynx.base_models import FunctionNode
from lightcurvelynx.effects.extinction import ExtinctionEffect
from lightcurvelynx.math_nodes.np_random import NumpyRandomFunc
from lightcurvelynx.math_nodes.ra_dec_sampler import ApproximateMOCSampler
from lightcurvelynx.math_nodes.scipy_random import SamplePDF
from lightcurvelynx.models.multi_object_model import RandomMultiObjectModel
from lightcurvelynx.models.sncosmo_models import SncosmoWrapperModel
from lightcurvelynx.noise_models.base_noise_models import PoissonFluxNoiseModel
from lightcurvelynx.obstable.lsst_obstable import LSSTObsTable
from lightcurvelynx.simulate import simulate_lightcurves
from lightcurvelynx.survey_info import SurveyInfo
from lightcurvelynx.utils.extrapolate import LinearDecayOnMag, ZeroPadding
from scipy.interpolate import interp1d

from hyrax_parsnip import RUBIN_DIA_SETTINGS

DP2_VISIT_DETECTOR_FILE = "/global/cfs/cdirs/lsst/shared/rubin/DP2/HATS/public-files/visit_detector.parquet"
DP2_DIA_CATALOG = "/global/cfs/cdirs/lsst/shared/rubin/DP2/HATS/dia_object_collection/dia_object_lc"

BANDS = list("ugrizy")
H0, OMEGA_M = 70.0, 0.315

# class -> simulation settings. CC classes list Vincenzi+19 template types and the
# peak absolute B magnitude (mean, sigma).
CLASSES = {
    "SNIa": {"zmax": 0.8},
    "SNII": {"zmax": 0.5, "types": ("SN II", "SN IIn", "SN IIb"), "m_abs": (-16.8, 0.8)},
    "SNIbc": {"zmax": 0.5, "types": ("SN Ib", "SN Ic", "SN Ic-BL"), "m_abs": (-17.5, 0.8)},
}

FLAG_COLUMNS = RUBIN_DIA_SETTINGS["flag_columns"]

EXTRAPOLATION = {
    "time_extrapolation": (ZeroPadding(), LinearDecayOnMag(decay_rate=0.02, mag_thres=30.0)),
    "wave_extrapolation": (ZeroPadding(), ZeroPadding()),
}


@dataclass
class Survey:
    survey_info: SurveyInfo
    footprint: object  # mocpy.MOC to draw positions from
    t_min: float
    t_max: float


def load_survey(visit_file=DP2_VISIT_DETECTOR_FILE, dia_catalog=DP2_DIA_CATALOG, noise_scale=1.5) -> Survey:
    """DP2 visits and noise model, and the area covered by both DP2 visits and the DIA catalog.

    ``noise_scale`` inflates the Poisson flux errors (DP2 errors are underestimated).
    """
    obstable = LSSTObsTable.from_ccdvisit_table(pd.read_parquet(visit_file), make_detector_footprint=True)
    passbands = PassbandGroup.from_preset("LSST", filters=BANDS)
    noise_model = PoissonFluxNoiseModel(err_scale=noise_scale)
    footprint = obstable.build_moc(max_depth=12).intersection(lsdb.open_catalog(dia_catalog).hc_structure.moc)
    return Survey(
        survey_info=SurveyInfo(obstable=obstable, passbands=passbands, noise_model=noise_model),
        footprint=footprint,
        t_min=float(obstable["time"].min()),
        t_max=float(obstable["time"].max()),
    )


def cc_rate(z):
    # Core-collapse rates follow the star-formation history, ~(1+z)^2.7 at low z.
    # Only the shape matters: it sets the redshift distribution.
    return (1 + z) ** 2.7


def redshift_node(label, zmax, rate):
    n, z = num_snia_per_redshift_bin(0.001, zmax, 100, H0=H0, Omega_m=OMEGA_M, vol_rate_function=rate)
    return SamplePDF(interp1d(z, n, bounds_error=False, fill_value=0), node_label=f"{label}_z")


def asymmetric_gaussian(mu, sigma_minus, sigma_plus):
    return lambda x: np.exp(-0.5 * ((x - mu) / np.where(x < mu, sigma_minus, sigma_plus)) ** 2)


def v19_templates(types):
    loaders = sncosmo.registry._get_registry(sncosmo.Source).get_loaders_metadata()
    return sorted(
        m["name"] for m in loaders if m["name"].startswith("v19") and m["name"].endswith("-corr") and m["type"] in types
    )


def amplitude_for(m_template):
    # At amplitude 1 a template peaks at B = m_template for distance modulus 0 (the sncosmo convention).
    def amplitude(m_abs, distmod):
        return 10 ** (-0.4 * (m_abs + distmod - m_template))

    return amplitude


def build_model(label, survey: Survey, zmax, types=None, m_abs=None):
    """Source model for one class. SN Ia without `types`; otherwise one random Vincenzi+19 template per object.

    Each class gets its own parameter nodes, labeled with the class name.
    """
    radec = ApproximateMOCSampler(survey.footprint, node_label=f"{label}_radec")
    z = redshift_node(label, zmax, snia_volumetric_rates if types is None else cc_rate)
    # Start t0 a month early to include SNe already fading when DP2 starts.
    t0 = NumpyRandomFunc("uniform", low=survey.t_min - 30, high=survey.t_max, node_label=f"{label}_t0")
    distmod = DistModFromRedshift(z, H0=H0, Omega_m=OMEGA_M, node_label=f"{label}_distmod")
    ebv = DustmapWrapper(SFDQuery(), ra=radec.ra, dec=radec.dec, node_label=f"{label}_mwext")
    extinction = ExtinctionEffect(extinction_model="F99", ebv=ebv, r_v=3.1, frame="observer", backend="dust_extinction")
    position = {"t0": t0, "redshift": z, "ra": radec.ra, "dec": radec.dec}

    if types is None:  # SN Ia: SALT3 with x0 from the Tripp relation
        x1 = SamplePDF(asymmetric_gaussian(0.973, 1.472, 0.222), node_label=f"{label}_x1")
        c = SamplePDF(asymmetric_gaussian(-0.054, 0.043, 0.101), node_label=f"{label}_c")
        m = NumpyRandomFunc("normal", loc=-19.3, scale=0.1, node_label=f"{label}_mabs")
        x0 = X0FromDistMod(distmod=distmod, x1=x1, c=c, alpha=0.15, beta=3.15, m_abs=m, node_label=f"{label}_x0")
        model = SncosmoWrapperModel("salt3", x0=x0, x1=x1, c=c, node_label=label, **position, **EXTRAPOLATION)
        model.add_effect(extinction)
        return model

    # Core collapse: one random template per object, scaled to the sampled peak magnitude.
    m = NumpyRandomFunc("normal", loc=m_abs[0], scale=m_abs[1], node_label=f"{label}_mabs")
    templates = []
    for name in v19_templates(types):
        m_template = sncosmo.get_source(name).peakmag("bessellb", "ab")
        amplitude = FunctionNode(amplitude_for(m_template), m_abs=m, distmod=distmod, node_label=f"{name}_amp")
        template = SncosmoWrapperModel(name, amplitude=amplitude, node_label=name, **position, **EXTRAPOLATION)
        # RandomMultiObjectModel can't apply observer-frame effects, so each template gets its own.
        template.add_effect(extinction)
        templates.append(template)
    return RandomMultiObjectModel(templates, node_label=label, **position)


def n_detections(flux, fluxerr, flagged, index, detection_snr):
    detected = (flux / fluxerr > detection_snr) & ~flagged
    return pd.Series(detected, index=index).groupby(level=0).sum()


def simulate_class(
    label, model, survey: Survey, n, rng, batch_size=2000, detection_snr=5.0, min_detections=2
) -> npd.NestedFrame:
    """Simulate batches until `n` objects have `min_detections` unsaturated points above `detection_snr`."""
    param_cols = [f"{label}_mwext.ebv"]
    if isinstance(model, RandomMultiObjectModel):
        param_cols.append(f"{label}.selected_object")

    kept, n_simulated = [], 0
    while sum(map(len, kept)) < n:
        lcs = simulate_lightcurves(
            model=model,
            num_samples=batch_size,
            survey_info=survey.survey_info,
            param_cols=param_cols,
            obstable_save_cols=["visitId"],
            rng=rng,
        )
        n_simulated += batch_size
        lcs = lcs.drop(columns="params").dropna(subset=["lightcurve"])
        flat = lcs["lightcurve"].nest.to_flat()
        n_det = n_detections(flat["flux"], flat["fluxerr"], flat["is_saturated"].astype(bool), flat.index, detection_snr)
        kept.append(lcs[n_det.reindex(lcs.index, fill_value=0).to_numpy() >= min_detections])

    lcs = pd.concat(kept, ignore_index=True).head(n)
    print(f"{label}: simulated {n_simulated:,}, kept {len(lcs):,} detected")
    return lcs


def to_dia_schema(lcs, label, first_id, detection_snr=5.0) -> npd.NestedFrame:
    """Rename lightcurvelynx output to the DP2 DIA schema, plus truth columns.

    lightcurvelynx fluxes are already in nJy, like Rubin's. Saturated points get
    ``psfDiffFlux_flag``; the other flags are all False.
    """
    flat = lcs["lightcurve"].nest.to_flat()
    sources = pd.DataFrame(
        {
            "band": flat["filter"].astype(str),
            "midpointMjdTai": flat["mjd"].astype(np.float64),
            "psfDiffFlux": flat["flux"].astype(np.float32),
            "psfDiffFluxErr": flat["fluxerr"].astype(np.float32),
            "visit": flat["visitId"].astype(np.int64),
        },
        index=flat.index,
    )
    for column in FLAG_COLUMNS:
        sources[column] = False
    sources["psfDiffFlux_flag"] = flat["is_saturated"].astype(bool)

    n_sources = n_detections(
        sources["psfDiffFlux"], sources["psfDiffFluxErr"], sources["psfDiffFlux_flag"], sources.index, detection_snr
    )
    template = lcs[f"{label}_selected_object"] if f"{label}_selected_object" in lcs.columns else "salt3"
    objects = pd.DataFrame(
        {
            "diaObjectId": np.arange(first_id, first_id + len(lcs), dtype=np.int64),
            "ra": lcs["ra"],
            "dec": lcs["dec"],
            "nDiaSources": n_sources.reindex(lcs.index, fill_value=0),
            "type": label,
            "redshift": lcs["z"],
            "t0": lcs["t0"],
            "mwebv": lcs[f"{label}_mwext_ebv"].astype(np.float64),
            "template": template,
        },
        index=lcs.index,
    )
    return npd.NestedFrame(objects).join_nested(sources, "diaObjectForcedSource")


def simulate_catalog(survey: Survey, n_per_class, classes=CLASSES, seed=None, **kwargs) -> npd.NestedFrame:
    """`n_per_class` detected objects of each class in the DP2 DIA schema. `kwargs` go to `simulate_class`."""
    rng = np.random.default_rng(seed)
    parts = []
    for label, settings in classes.items():
        lcs = simulate_class(label, build_model(label, survey, **settings), survey, n_per_class, rng, **kwargs)
        parts.append(to_dia_schema(lcs, label, sum(map(len, parts)), kwargs.get("detection_snr", 5.0)))
    return pd.concat(parts, ignore_index=True)


def write_catalog(frame, path, catalog_name="dp2_sim_sne"):
    """Write as a HATS collection (catalog + margin), like the DP2 dia_object_collection."""
    lsdb.from_dataframe(frame, ra_column="ra", dec_column="dec", catalog_name=catalog_name).write_catalog(
        path, overwrite=True
    )


def flat_schema(catalog_path) -> dict:
    """Column name -> pyarrow type, with nested columns as "nested.column"."""
    fields = {}
    for field in lsdb.open_catalog(str(catalog_path)).hc_structure.schema:
        if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):  # nested: list<struct>
            fields.update({f"{field.name}.{child.name}": child.type for child in field.type.value_type})
        else:
            fields[field.name] = field.type
    return fields


def check_schema(catalog_path, real_catalog=DP2_DIA_CATALOG) -> list[str]:
    """Check the simulated catalog against the real DIA catalog; return the simulation-only columns.

    Columns in both must have the same type, and both must have every column
    `RUBIN_DIA_SETTINGS` reads.
    """
    sim, real = flat_schema(catalog_path), flat_schema(real_catalog)
    mismatched = {k: (v, real[k]) for k, v in sim.items() if k in real and real[k] != v}
    if mismatched:
        raise ValueError(f"Column types differ from {real_catalog}: {mismatched}")

    s = RUBIN_DIA_SETTINGS
    nested = [s[k] for k in ("time_column", "flux_column", "fluxerr_column", "band_column")] + s["flag_columns"]
    used = [s["id_column"]] + [f"{s['lightcurve_column']}.{c}" for c in nested]
    missing = [c for c in used if c not in sim or c not in real]
    if missing:
        raise ValueError(f"Columns used by RUBIN_DIA_SETTINGS are missing: {missing}")
    return sorted(set(sim) - set(real))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", nargs="?", default="data/dp2_sim_sne", help="Output HATS catalog")
    parser.add_argument("--n-per-class", type=int, default=1000, help="Detected objects kept per class")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2000, help="Objects per simulate_lightcurves call")
    parser.add_argument("--noise-scale", type=float, default=1.5, help="Inflate Poisson flux errors by this factor")
    parser.add_argument("--detection-snr", type=float, default=5.0)
    parser.add_argument("--min-detections", type=int, default=2)
    parser.add_argument("--visit-file", default=DP2_VISIT_DETECTOR_FILE, help="DP2 visit-detector parquet")
    parser.add_argument("--dia-catalog", default=DP2_DIA_CATALOG, help="Real DIA catalog for footprint and schema")
    args = parser.parse_args()

    survey = load_survey(args.visit_file, args.dia_catalog, noise_scale=args.noise_scale)
    frame = simulate_catalog(
        survey,
        args.n_per_class,
        seed=args.seed,
        batch_size=args.batch_size,
        detection_snr=args.detection_snr,
        min_detections=args.min_detections,
    )
    write_catalog(frame, args.output)
    extra = check_schema(args.output, args.dia_catalog)
    print(f"Wrote {len(frame):,} objects to {args.output} (simulation-only columns: {extra})")


if __name__ == "__main__":
    main()
