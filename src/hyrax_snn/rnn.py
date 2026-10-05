"""SuperNNova's vanilla RNN classifier.

Adapted from ``supernnova/training/vanilla_rnn.py`` (SuperNNova, MIT License,
https://github.com/supernnova/SuperNNova). The layer names are kept, so state dicts
saved by SuperNNova (e.g. the Fink models) load unchanged. The constructor takes the
hyperparameters directly instead of SuperNNova's ``ExperimentSettings``.
"""

import torch


class VanillaRNN(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        nb_classes: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.05,
        bidirectional: bool = True,
        layer_type: str = "lstm",
        rnn_output_option: str = "mean",
    ):
        super().__init__()
        self.layer_type = layer_type
        self.rnn_output_option = rnn_output_option

        bidirectional_factor = 2 if bidirectional else 1
        last_input_size = (
            hidden_dim * bidirectional_factor
            if rnn_output_option == "mean"
            else hidden_dim * bidirectional_factor * num_layers
        )

        self.rnn_layer = getattr(torch.nn, layer_type.upper())(
            input_size,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.output_dropout_layer = torch.nn.Dropout(dropout)
        self.output_layer = torch.nn.Linear(last_input_size, nb_classes)

    def forward(self, x):
        """Class logits for a (packed or padded, sequence-first) batch of light curves."""
        x, hidden = self.rnn_layer(x)

        if self.rnn_output_option == "standard":
            hn = hidden[0] if self.layer_type == "lstm" else hidden
            # (num_layers * num_directions, batch, hidden) -> (batch, hidden * num_layers * num_directions)
            hn = hn.permute(1, 2, 0).contiguous()
            x = hn.view(hn.shape[0], -1)

        if self.rnn_output_option == "mean":
            if isinstance(x, torch.nn.utils.rnn.PackedSequence):
                x, lens = torch.nn.utils.rnn.pad_packed_sequence(x)
                # Mean over each sequence's real (unpadded) time steps.
                x = x.sum(0) / lens.unsqueeze(-1).float().to(x.device)
            else:
                x = x.mean(0)

        x = self.output_dropout_layer(x)
        return self.output_layer(x)
