"""Q-learning trainer and training-time validation for GSP.

The online network predicts Q(s, a).  The target network supplies the
Bellman bootstrap, with illegal actions masked out.  Search-derived finite
returns are combined with the bootstrap through ``y = max(R, y_hat)``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from environment import MazeInstance
from model import build_q_network
from search import QGuidedWAStar, SearchResult


CATEGORY_UNSOLVED = "unsolved"
CATEGORY_SOLVED = "solved"
CATEGORY_SATISFICED = "satisficed"
INSTANCE_CATEGORIES = (CATEGORY_UNSOLVED, CATEGORY_SOLVED, CATEGORY_SATISFICED)


@dataclass
class UpdateStats:
    """Diagnostics returned by :meth:`QTrainer.train_step`."""

    loss: float
    q_mean: float
    q_min: float
    q_max: float
    gradient_norm: float
    target_network_updated: bool
    gradient_updates: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "loss": self.loss,
            "q_mean": self.q_mean,
            "q_min": self.q_min,
            "q_max": self.q_max,
            "gradient_norm": self.gradient_norm,
            "target_network_updated": self.target_network_updated,
            "gradient_updates": self.gradient_updates,
        }


@dataclass
class ValidationMetrics:
    """Aggregate validation metrics; no BFS labels are used."""

    num_instances: int
    num_solved: int
    success_rate: float
    mean_expansions_all: float
    mean_expansions_solved: float
    mean_plan_length_solved: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "num_instances": self.num_instances,
            "num_solved": self.num_solved,
            "success_rate": self.success_rate,
            "mean_expansions_all": self.mean_expansions_all,
            "mean_expansions_solved": (
                self.mean_expansions_solved if self.num_solved > 0 else None
            ),
            "mean_plan_length_solved": (
                self.mean_plan_length_solved if self.num_solved > 0 else None
            ),
        }

    def selection_key(self) -> tuple:
        """Higher is better: success, fewer solved expansions, shorter paths."""
        expansions = (
            self.mean_expansions_solved
            if self.num_solved > 0
            else float("inf")
        )
        plan = self.mean_plan_length_solved if self.num_solved > 0 else float("inf")
        return (self.success_rate, -expansions, -plan)


class QTrainer:
    """Holds the online/target networks and performs Bellman updates on CUDA."""

    REQUIRED_BATCH_KEYS = (
        "states",
        "actions",
        "next_states",
        "return_bounds",
        "terminal",
        "dead_end",
        "next_valid_mask",
    )

    def __init__(
        self,
        config: Config,
        *,
        device: Optional[str | torch.device] = None,
        model: Optional[nn.Module] = None,
        target_model: Optional[nn.Module] = None,
    ) -> None:
        self.config = config
        selected = torch.device(config.device if device is None else device)
        if selected.type != "cuda":
            raise RuntimeError(
                "QTrainer refuses non-CUDA devices. Formal training must use an NVIDIA GPU."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Install a CUDA-enabled PyTorch build and NVIDIA "
                "drivers before starting GSP training."
            )
        if selected.index is not None and selected.index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested CUDA device does not exist: {selected}")
        self.device = selected

        self.model = model if model is not None else build_q_network(config)
        self.model.to(self.device)
        if target_model is None:
            self.target_model = build_q_network(config)
        else:
            self.target_model = target_model
        self.target_model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.0,
        )
        self.loss_function = nn.MSELoss()
        self.gradient_updates = 0
        self.target_updates = 0
        self._verify_cuda_model()

    def _verify_cuda_model(self) -> None:
        parameter_devices = [parameter.device.type for parameter in self.model.parameters()]
        target_devices = [parameter.device.type for parameter in self.target_model.parameters()]
        if any(device != "cuda" for device in parameter_devices + target_devices):
            raise RuntimeError("Both Q networks must reside entirely on CUDA")
        if next(self.model.parameters()).device != self.device:
            raise RuntimeError(
                f"Model is on {next(self.model.parameters()).device}, expected {self.device}"
            )

    def _validate_batch(self, batch: Mapping[str, torch.Tensor]) -> None:
        missing = [key for key in self.REQUIRED_BATCH_KEYS if key not in batch]
        if missing:
            raise KeyError(f"Training batch is missing keys: {missing}")
        for key in self.REQUIRED_BATCH_KEYS:
            tensor = batch[key]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Batch entry '{key}' must be a tensor")
            if tensor.device.type != "cuda" or tensor.device != self.device:
                raise RuntimeError(
                    f"Batch entry '{key}' is on {tensor.device}, not required CUDA device "
                    f"{self.device}"
                )
        if batch["states"].shape != batch["next_states"].shape:
            raise ValueError("states and next_states must have identical batch shapes")
        if batch["states"].shape[1] != self.config.state_dim:
            raise ValueError(f"Expected state dimension {self.config.state_dim}")
        if batch["actions"].shape[0] != batch["states"].shape[0]:
            raise ValueError("actions batch length mismatch")
        if batch["next_valid_mask"].shape != (batch["states"].shape[0], 4):
            raise ValueError("next_valid_mask must have shape (batch, 4)")
        if batch["states"].shape[0] == 0:
            raise ValueError("Empty batch")
        if not torch.all((batch["actions"] >= 0) & (batch["actions"] < 4)):
            raise ValueError("Action indices must be in [0, 3]")

    def compute_targets(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Construct the final target values for a batch of transitions.

        This method is public so target semantics can be tested explicitly.
        It never updates either network.
        """
        self._validate_batch(batch)
        self._verify_cuda_model()
        self.target_model.eval()
        with torch.no_grad():
            next_q = self.target_model(batch["next_states"].float())
            legal_mask = batch["next_valid_mask"].bool()
            legal_any = legal_mask.any(dim=1)
            active_bootstrap = (~batch["terminal"]) & (~batch["dead_end"])
            if bool((active_bootstrap & (~legal_any)).any()):
                raise RuntimeError(
                    "A non-terminal, non-dead-end transition has no legal successor actions"
                )
            masked_q = next_q.masked_fill(~legal_mask, -float("inf"))
            max_next_q = masked_q.max(dim=1).values
            # Rows with no legal action are terminal/dead ends and are replaced
            # below; zero keeps their intermediate value finite.
            max_next_q = torch.where(legal_any, max_next_q, torch.zeros_like(max_next_q))
            bellman_hat = self.config.step_reward + self.config.gamma * max_next_q

            if self.config.use_solution_bounds:
                finite_bound = torch.isfinite(batch["return_bounds"])
                target = torch.where(
                    finite_bound,
                    torch.maximum(batch["return_bounds"], bellman_hat),
                    bellman_hat,
                )
            else:
                target = bellman_hat

            target = torch.where(
                batch["terminal"],
                torch.full_like(target, float(self.config.step_reward)),
                target,
            )
            target = torch.where(
                batch["dead_end"],
                torch.full_like(target, float(self.config.dead_end_return)),
                target,
            )
        return target

    def train_step(self, batch: Mapping[str, torch.Tensor]) -> UpdateStats:
        """Perform one Q-learning gradient update and return diagnostics."""
        self._validate_batch(batch)
        self._verify_cuda_model()
        self.model.train()
        self.target_model.eval()

        target = self.compute_targets(batch)

        self.optimizer.zero_grad(set_to_none=True)
        q_values = self.model.get_q_values(batch["states"].float())
        if q_values.shape != (batch["states"].shape[0], 4):
            raise RuntimeError(f"Unexpected Q output shape: {tuple(q_values.shape)}")
        chosen_q = q_values.gather(1, batch["actions"].long().unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(chosen_q, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")

        detached_q = q_values.detach()
        q_mean = float(detached_q.mean().item())
        q_min = float(detached_q.min().item())
        q_max = float(detached_q.max().item())

        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.gradient_clip_norm,
            ).item()
        )
        self.optimizer.step()

        self.gradient_updates += 1
        target_updated = False
        if self.gradient_updates % self.config.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
            for parameter in self.target_model.parameters():
                parameter.requires_grad_(False)
            self.target_updates += 1
            target_updated = True

        return UpdateStats(
            loss=float(loss.item()),
            q_mean=q_mean,
            q_min=q_min,
            q_max=q_max,
            gradient_norm=gradient_norm,
            target_network_updated=target_updated,
            gradient_updates=self.gradient_updates,
        )

    def full_batch_device_report(self, batch: Mapping[str, torch.Tensor]) -> Dict[str, str]:
        return {
            "model_parameters": str(next(self.model.parameters()).device),
            "target_parameters": str(next(self.target_model.parameters()).device),
            "batch_states": str(batch["states"].device),
            "batch_targets": str(batch["actions"].device),
        }


class InstancePoolSampler:
    """Weighted sampling from mutually exclusive search-outcome pools."""

    def __init__(self, config: Config, instance_ids: Sequence[int], seed: int) -> None:
        if len(instance_ids) == 0:
            raise ValueError("At least one training instance is required")
        self.config = config
        self.instance_ids = [int(i) for i in instance_ids]
        if len(set(self.instance_ids)) != len(self.instance_ids):
            raise ValueError("Instance ids must be unique")
        self._pools: Dict[str, Set[int]] = {name: set() for name in INSTANCE_CATEGORIES}
        self._pools[CATEGORY_UNSOLVED] = set(self.instance_ids)
        self._rng = random.Random(int(seed))
        self._weights = {
            CATEGORY_UNSOLVED: float(config.unsolved_weight),
            CATEGORY_SOLVED: float(config.solved_weight),
            CATEGORY_SATISFICED: float(config.satisficed_weight),
        }

    @staticmethod
    def classify(result: SearchResult) -> str:
        if not result.success:
            return CATEGORY_UNSOLVED
        if result.expansions == result.plan_length:
            return CATEGORY_SOLVED
        return CATEGORY_SATISFICED

    def update(self, instance_id: int, category: str) -> None:
        instance_id = int(instance_id)
        if category not in INSTANCE_CATEGORIES:
            raise ValueError(f"Unknown instance category: {category}")
        if instance_id not in self._pools[category] and not any(
            instance_id in pool for pool in self._pools.values()
        ):
            raise ValueError(f"Unknown instance id: {instance_id}")
        for pool in self._pools.values():
            pool.discard(instance_id)
        self._pools[category].add(instance_id)

    def choose(self) -> int:
        if not self.config.use_instance_sampling:
            return self._rng.choice(self.instance_ids)
        available = [
            (name, self._weights[name], sorted(pool))
            for name, pool in self._pools.items()
            if pool and self._weights[name] > 0
        ]
        if not available:
            raise RuntimeError("All instance pools are empty or have zero sampling weight")
        total = sum(weight for _, weight, _ in available)
        draw = self._rng.random() * total
        cumulative = 0.0
        selected = available[-1]
        for candidate in available:
            cumulative += candidate[1]
            if draw <= cumulative:
                selected = candidate
                break
        _, _, pool = selected
        return self._rng.choice(pool)

    def counts(self) -> Dict[str, int]:
        return {name: len(pool) for name, pool in self._pools.items()}

    def category_of(self, instance_id: int) -> str:
        for name, pool in self._pools.items():
            if int(instance_id) in pool:
                return name
        raise KeyError(f"Instance {instance_id} is not in any pool")


@torch.no_grad()
def validate_network(
    network: nn.Module,
    instances: Sequence[MazeInstance],
    config: Config,
    device: str | torch.device,
) -> ValidationMetrics:
    """Evaluate the current Q-guided WA* on validation mazes.

    No gradients, replay writes, pool updates, or parameter updates occur.
    """
    if len(instances) == 0:
        raise ValueError("Validation requires at least one instance")
    search = QGuidedWAStar(network, config, device)
    solved = 0
    all_expansions = 0
    solved_expansions = 0
    solved_plan_lengths = 0
    network.eval()
    for instance in instances:
        result = search.search_instance(instance)
        all_expansions += result.expansions
        if result.success:
            solved += 1
            solved_expansions += result.expansions
            solved_plan_lengths += result.plan_length
    count = len(instances)
    return ValidationMetrics(
        num_instances=count,
        num_solved=solved,
        success_rate=solved / count,
        mean_expansions_all=all_expansions / count,
        mean_expansions_solved=(solved_expansions / solved) if solved else float("nan"),
        mean_plan_length_solved=(solved_plan_lengths / solved) if solved else float("nan"),
    )