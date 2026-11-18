"""MLP Q-network used as the learned heuristic inside WA*."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import Config


class QNetwork(nn.Module):
    """Fully connected Q function: 192 -> 128 -> 128 -> 4 by default.

    No Dropout, BatchNorm or weight decay is used.  The output layer is
    linear because Q values are unconstrained regression targets.
    """

    def __init__(
        self,
        input_dim: int = 192,
        hidden_dims: Sequence[int] = (128, 128),
        output_dim: int = 4,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if len(hidden_dims) == 0:
            raise ValueError("At least one hidden layer is required")
        self.input_dim = int(input_dim)
        self.hidden_dims: Tuple[int, ...] = tuple(int(v) for v in hidden_dims)
        self.output_dim = int(output_dim)

        layers = []
        previous = self.input_dim
        for hidden in self.hidden_dims:
            if hidden <= 0:
                raise ValueError("Hidden layer sizes must be positive")
            layers.append(nn.Linear(previous, hidden))
            layers.append(nn.ReLU())
            previous = hidden
        layers.append(nn.Linear(previous, self.output_dim))
        self.network = nn.Sequential(*layers)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        linear_layers = [module for module in self.network if isinstance(module, nn.Linear)]
        hidden_layers = linear_layers[:-1]
        output_layer = linear_layers[-1]
        for layer in hidden_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        # Small output scale reduces the initial Q magnitude and stabilises
        # target-network bootstrapping in this small-data regime.
        nn.init.xavier_uniform_(output_layer.weight, gain=0.1)
        nn.init.zeros_(output_layer.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Map states ``(..., input_dim)`` to Q values ``(..., 4)``."""
        if state.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected state dimension {self.input_dim}, got shape {tuple(state.shape)}"
            )
        return self.network(state)

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Explicit inference alias required by the project interface."""
        return self.forward(state)


def build_q_network(
    config: Config,
    *,
    device: str | torch.device | None = None,
) -> QNetwork:
    """Create a Q-network matching ``config`` and optionally move it."""
    network = QNetwork(
        input_dim=config.state_dim,
        hidden_dims=config.hidden_dims,
        output_dim=config.output_dim,
    )
    if device is not None:
        network = network.to(torch.device(device))
    return network


@torch.no_grad()
def predict_q_values(
    network: QNetwork,
    states: np.ndarray | torch.Tensor,
    device: str | torch.device,
) -> np.ndarray:
    """Convenience NumPy inference helper for one state or a batch."""
    tensor = torch.as_tensor(states, dtype=torch.float32, device=torch.device(device))
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    values = network.get_q_values(tensor)
    return values.detach().cpu().numpy()