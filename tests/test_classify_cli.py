"""Catalog options of examples/classify_rubin_dia.py."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "examples" / "classify_rubin_dia.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("classify_rubin_dia", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settings_for(cli, monkeypatch, *options):
    monkeypatch.setattr(sys, "argv", ["classify_rubin_dia.py", "catalog.hats", "--method", "parsnip", *options])
    from hyrax_lightcurves.config import RUBIN_DIA_SETTINGS

    return cli.catalog_settings(RUBIN_DIA_SETTINGS, cli.parse_args())


def test_defaults_unchanged(cli, monkeypatch):
    from hyrax_lightcurves.config import RUBIN_DIA_SETTINGS

    assert settings_for(cli, monkeypatch) == RUBIN_DIA_SETTINGS


def test_flag_columns_without_names_means_no_flags(cli, monkeypatch):
    assert settings_for(cli, monkeypatch, "--flag-columns")["flag_columns"] == []
    assert settings_for(cli, monkeypatch, "--flag-columns", "a", "b")["flag_columns"] == ["a", "b"]


def test_lightcurve_and_time_columns(cli, monkeypatch):
    settings = settings_for(cli, monkeypatch, "--lightcurve-column", "lc_nightly", "--time-column", "t")
    assert settings["lightcurve_column"] == "lc_nightly"
    assert settings["time_column"] == "t"
