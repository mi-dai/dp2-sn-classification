"""SCONE's CNN in PyTorch, ported from helenqu/scone ``SconeClassifier._define_and_compile_model`` (Keras).

Keras works in (batch, height, width, channels) and SCONE's heatmaps are (wavelength, time, 2);
here they are (batch, 2, wavelength, time). Each convolution spans all wavelength rows of its input
and SCONE's custom "Reshape" layer (transpose [0, 3, 2, 1]) then turns the filters back into the
wavelength-like rows of a 1-channel image: ``permute(0, 2, 1, 3)`` in NCHW. Layer sizes, activations,
dropout, the Keras BatchNormalization defaults (momentum 0.99, epsilon 1e-3) and Glorot-uniform
initialization follow upstream.

One difference: the output is one logit per class (softmax). Upstream's binary model has a single
sigmoid unit, which is the same function of the features (a 2-class softmax depends only on the
difference of the two logits); one code path serves any number of classes.
"""

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """ZeroPadding2D((0, 1)) → Conv2D(filters, (rows, 3), elu) → SCONE's Reshape → BatchNormalization."""

    def __init__(self, in_channels: int, rows: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, rows, kernel_size=(rows, 3))
        # Keras momentum 0.99 = PyTorch momentum 0.01.
        self.norm = nn.BatchNorm2d(1, eps=1e-3, momentum=0.01)

    def forward(self, x):
        x = nn.functional.pad(x, (1, 1))  # one zero column on each side of the time axis
        x = nn.functional.elu(self.conv(x))  # (batch, rows, 1, time)
        return self.norm(x.permute(0, 2, 1, 3))  # (batch, 1, rows, time)


class SconeCNN(nn.Module):
    """SCONE's classifier for (batch, 2, n_wavelengths, n_times) heatmaps.

    Parameters
    ----------
    n_classes : int
        Number of output classes (logits).
    n_wavelengths, n_times : int
        Heatmap size (SCONE: 32 × 180). Both must be divisible by 4.
    with_z : bool
        Whether the redshift and its error are concatenated to the image features (SCONE's ``with_z``).
    """

    def __init__(self, n_classes: int, n_wavelengths: int = 32, n_times: int = 180, with_z: bool = False):
        super().__init__()
        if n_wavelengths % 4 or n_times % 4:
            raise ValueError(f"Heatmap size {n_wavelengths} x {n_times} must be divisible by 4")
        rows = n_wavelengths
        self.with_z = with_z
        self.features = nn.Sequential(
            _ConvBlock(2, rows),
            _ConvBlock(1, rows),
            nn.MaxPool2d(2),
            _ConvBlock(1, rows // 2),
            _ConvBlock(1, rows // 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.5),
        )
        n_features = (rows // 4) * (n_times // 4) + (2 if with_z else 0)
        self.dense = nn.Linear(n_features, 32)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(32, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        """Keras defaults: Glorot-uniform kernels, zero biases."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, heatmap, redshift=None, redshift_error=None):
        x = self.features(heatmap)
        if self.with_z:
            x = torch.cat([x, redshift.reshape(-1, 1), redshift_error.reshape(-1, 1)], dim=1)
        x = self.dropout(nn.functional.relu(self.dense(x)))
        return self.output(x)
