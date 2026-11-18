"""Central configuration for the GSP reproduction project.

All experiment-defining hyper-parameters live here.  Other modules must read
them from a :class:`Config` instance instead of duplicating the values.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass
class Config:
    """Experiment configuration.

    The defaults correspond to the setup described in the project brief.
    Changing the grid size automatically changes the state dimension
    (``height * width * 3``); the default 8 x 8 grid therefore gives 192.
    """

    # ------------------------------------------------------------------
    # Grid environment
    # ------------------------------------------------------------------
    height: int = 8
    width: int = 8
    min_obstacle_ratio: float = 0.10
    max_obstacle_ratio: float = 0.25
    num_actions: int = 4
    step_reward: float = -1.0

    # ------------------------------------------------------------------
    # Q network
    # ------------------------------------------------------------------
    hidden_dims: Tuple[int, ...] = (128, 128)
    output_dim: int = 4

    # ------------------------------------------------------------------
    # Weighted A*
    # ------------------------------------------------------------------
    wa_star_weight: float = 2.0
    search_budget: int = 128
    dead_end_return: float = -64.0

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------
    num_train_instances: int = 1000
    num_validation_instances: int = 200
    num_test_instances: int = 300
    train_dataset_seed: int = 20240101
    validation_dataset_seed: int = 20240202
    test_dataset_seed: int = 20240303
    min_bfs_length: int = 3
    easy_max_bfs_length: int = 6
    medium_max_bfs_length: int = 10
    # Per-target-difficulty attempt cap.  The generator never loops forever.
    max_generation_attempts_per_target: int = 200
    # Cap used when filling any missing samples after stratification.
    max_generation_attempts_per_sample: int = 100

    # ------------------------------------------------------------------
    # Reinforcement-learning / training
    # ------------------------------------------------------------------
    training_seed: int = 42
    training_seeds: Tuple[int, ...] = (42, 43, 44)
    episodes: int = 3000
    replay_capacity: int = 20000
    batch_size: int = 64
    updates_per_episode: int = 2
    learning_rate: float = 1e-3
    gamma: float = 1.0
    target_update_interval: int = 200
    gradient_clip_norm: float = 1.0
    validation_interval: int = 100

    # Dynamic instance sampling weights (simplified initial setting).
    unsolved_weight: float = 1.0
    solved_weight: float = 2.0
    satisficed_weight: float = 4.0
    use_instance_sampling: bool = True
    use_solution_bounds: bool = True

    # ------------------------------------------------------------------
    # Runtime / outputs
    # ------------------------------------------------------------------
    device: str = "cuda:0"
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"
    figures_dir: str = "results/figures"
    include_greedy_evaluation: bool = False
    greedy_max_steps: int = 128

    def __post_init__(self) -> None:
        # Accept lists loaded from JSON as tuples.
        self.hidden_dims = tuple(int(v) for v in self.hidden_dims)
        self.training_seeds = tuple(int(v) for v in self.training_seeds)
        self.validate()

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def cell_count(self) -> int:
        return self.height * self.width

    @property
    def state_dim(self) -> int:
        """Dimension of ``[M, P, G]``: three 8x8 channels by default."""
        return 3 * self.height * self.width

    @property
    def gpu_index(self) -> int:
        """CUDA device index parsed from ``device`` (e.g. cuda:0 -> 0)."""
        if ":" not in self.device:
            return 0
        try:
            return int(self.device.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device string: {self.device}") from exc

    @property
    def train_path(self) -> Path:
        return Path(self.data_dir) / "train.npz"

    @property
    def validation_path(self) -> Path:
        return Path(self.data_dir) / "validation.npz"

    @property
    def test_path(self) -> Path:
        return Path(self.data_dir) / "test.npz"

    @property
    def manifest_path(self) -> Path:
        return Path(self.data_dir) / "dataset_manifest.json"

    @property
    def best_checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir) / "best_model.pt"

    @property
    def last_checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir) / "last_model.pt"

    @property
    def training_log_path(self) -> Path:
        return Path(self.log_dir) / "training_log.csv"

    @property
    def validation_log_path(self) -> Path:
        return Path(self.log_dir) / "validation_log.csv"

    @property
    def evaluation_csv_path(self) -> Path:
        return Path(self.results_dir) / "evaluation.csv"

    @property
    def summary_path(self) -> Path:
        return Path(self.results_dir) / "summary.json"

    # ------------------------------------------------------------------
    # Validation and serialization
    # ------------------------------------------------------------------
    def validate(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Grid dimensions must be positive")
        if self.num_actions != 4 or self.output_dim != 4:
            raise ValueError("This grid-world implementation requires four actions")
        if not (0.0 <= self.min_obstacle_ratio <= self.max_obstacle_ratio < 1.0):
            raise ValueError("Require 0 <= min_obstacle_ratio <= max_obstacle_ratio < 1")
        if self.state_dim != self.height * self.width * 3:
            raise ValueError("Inconsistent state dimension")
        if len(self.hidden_dims) == 0 or any(v <= 0 for v in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        if self.output_dim != self.num_actions:
            raise ValueError("Network output must equal the number of actions")
        if self.search_budget <= 0:
            raise ValueError("search_budget must be positive")
        if self.wa_star_weight <= 0:
            raise ValueError("wa_star_weight must be positive")
        if self.num_train_instances <= 0 or self.num_validation_instances <= 0 or self.num_test_instances <= 0:
            raise ValueError("All dataset sizes must be positive")
        if len({self.train_dataset_seed, self.validation_dataset_seed, self.test_dataset_seed}) != 3:
            raise ValueError("Train/validation/test dataset seeds must be distinct")
        if self.min_bfs_length < 1:
            raise ValueError("min_bfs_length must be >= 1")
        if not (self.min_bfs_length <= self.easy_max_bfs_length < self.medium_max_bfs_length):
            raise ValueError("Difficulty thresholds must satisfy min <= easy < medium")
        if self.episodes <= 0 or self.batch_size <= 0 or self.replay_capacity < self.batch_size:
            raise ValueError("Invalid episode/batch/replay configuration")
        if self.updates_per_episode < 0:
            raise ValueError("updates_per_episode must be non-negative")
        if self.updates_per_episode > 0 and self.gamma < 0:
            raise ValueError("gamma must be non-negative")
        if self.target_update_interval <= 0:
            raise ValueError("target_update_interval must be positive")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.validation_interval <= 0:
            raise ValueError("validation_interval must be positive")
        if self.min_bfs_length > self.height * self.width:
            raise ValueError("min_bfs_length is larger than the grid")
        if self.step_reward != -1.0:
            raise ValueError("This GSP reproduction fixes the per-move reward at -1")
        if not self.device.startswith("cuda"):
            raise ValueError("Formal GSP training/evaluation requires a CUDA device")
        if any(w < 0 for w in (self.unsolved_weight, self.solved_weight, self.satisficed_weight)):
            raise ValueError("Instance-pool weights must be non-negative")
        if sum((self.unsolved_weight, self.solved_weight, self.satisficed_weight)) <= 0:
            raise ValueError("At least one instance-pool weight must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        result = dataclasses.asdict(self)
        result["hidden_dims"] = list(self.hidden_dims)
        result["training_seeds"] = list(self.training_seeds)
        return result

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in values.items() if k in known}
        if "hidden_dims" in filtered:
            filtered["hidden_dims"] = tuple(filtered["hidden_dims"])
        if "training_seeds" in filtered:
            filtered["training_seeds"] = tuple(filtered["training_seeds"])
        return cls(**filtered)

    def copy_with(self, **overrides: Any) -> "Config":
        """Return a validated copy with selected fields replaced."""
        return dataclasses.replace(self, **overrides)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> "Config":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


DEFAULT_CONFIG = Config()