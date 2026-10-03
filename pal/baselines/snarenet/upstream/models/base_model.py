import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

import operator
from functools import reduce


class BaseModel(nn.Module):
    """
    Shared feedforward network architecture used by all constraint-enforcement models.
    Hidden layers with BatchNorm, ReLU, Dropout, and Kaiming-initialized weights.

    The number of hidden layers is read from `cfg.model.num_hidden_layers` (default 2, as upstream).
    """

    def __init__(self, data, cfg, output_dim=None):
        """
        Args:
            data: Problem data object providing encoded_xdim and ydim.
            cfg: Config object with cfg.model.hidden_size, cfg.model.num_hidden_layers.
            output_dim: Output dimension override. Defaults to data.ydim.
                        DC3 passes ydim - neq when using partial completion.
        """
        super().__init__()
        if output_dim is None:
            output_dim = data.ydim

        # Configurable depth, upstream hardcodes 2 hidden layers.
        num_hidden = getattr(cfg.model, "num_hidden_layers", 2)
        layer_sizes = [data.encoded_xdim] + [cfg.model.hidden_size] * num_hidden

        # BatchNorm and Dropout cause unstable training for CBF dataset. Exclude them in base model for CBFs.
        if cfg.dataset.prob_type == 'cbf':
            layers = reduce(operator.add,
                [[nn.Linear(a, b), nn.ReLU()]
                    for a, b in zip(layer_sizes[:-1], layer_sizes[1:])])
        # All other datasets use BatchNorm and Dropout in the base model.
        else:
            layers = reduce(operator.add,
                [[nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)]
                    for a, b in zip(layer_sizes[:-1], layer_sizes[1:])])
        
        layers += [nn.Linear(layer_sizes[-1], output_dim)]

        # Kaiming initialization for all linear layers
        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)

        self.net = nn.Sequential(*layers)

    def forward(self, encoded_x):
        return self.net(encoded_x)
