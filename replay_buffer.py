"""FIFO replay buffer holding state-action transitions from many searches."""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np
import torch

from search import TransitionExperience


class ReplayBuffer:
    """Bounded FIFO experience buffer with uniform random mini-batches."""

    def __init__(self, capacity: int, *, seed: Optional[int] = None) -> None:
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive")
        self.capacity = int(capacity)
        self._buffer: Deque[TransitionExperience] = deque(maxlen=self.capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, experience: TransitionExperience) -> None:
        if not isinstance(experience, TransitionExperience):
            raise TypeError("ReplayBuffer accepts TransitionExperience objects")
        self._buffer.append(experience)

    def add_many(self, experiences: Sequence[TransitionExperience]) -> int:
        count = 0
        for experience in experiences:
            self.add(experience)
            count += 1
        return count

    def clear(self) -> None:
        self._buffer.clear()

    def get_all(self) -> List[TransitionExperience]:
        """Return a shallow snapshot (useful for diagnostics/tests)."""
        return list(self._buffer)

    def sample_experiences(self, batch_size: int) -> List[TransitionExperience]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer of size {len(self._buffer)}"
            )
        indices = self._rng.sample(range(len(self._buffer)), batch_size)
        return [self._buffer[index] for index in indices]

    def sample(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> Dict[str, torch.Tensor]:
        """Sample and collate a batch of tensors onto ``device``."""
        experiences = self.sample_experiences(batch_size)
        states = np.stack([experience.state for experience in experiences], axis=0)
        actions = np.asarray([experience.action for experience in experiences], dtype=np.int64)
        next_states = np.stack([experience.next_state for experience in experiences], axis=0)
        return_bounds = np.asarray(
            [experience.return_bound for experience in experiences], dtype=np.float32
        )
        terminal = np.asarray([experience.terminal for experience in experiences], dtype=np.bool_)
        dead_end = np.asarray([experience.dead_end for experience in experiences], dtype=np.bool_)
        next_valid_mask = np.stack(
            [experience.next_valid_mask for experience in experiences], axis=0
        ).astype(np.bool_)
        target_device = torch.device(device)
        return {
            "states": torch.as_tensor(states, dtype=torch.float32, device=target_device),
            "actions": torch.as_tensor(actions, dtype=torch.long, device=target_device),
            "next_states": torch.as_tensor(next_states, dtype=torch.float32, device=target_device),
            "return_bounds": torch.as_tensor(
                return_bounds, dtype=torch.float32, device=target_device
            ),
            "terminal": torch.as_tensor(terminal, dtype=torch.bool, device=target_device),
            "dead_end": torch.as_tensor(dead_end, dtype=torch.bool, device=target_device),
            "next_valid_mask": torch.as_tensor(
                next_valid_mask, dtype=torch.bool, device=target_device
            ),
        }

    def stats(self) -> Dict[str, int]:
        terminal_count = sum(1 for exp in self._buffer if exp.terminal)
        dead_end_count = sum(1 for exp in self._buffer if exp.dead_end)
        finite_bound_count = sum(
            1 for exp in self._buffer if np.isfinite(exp.return_bound) and not exp.dead_end
        )
        return {
            "size": len(self._buffer),
            "capacity": self.capacity,
            "terminal": terminal_count,
            "dead_end": dead_end_count,
            "finite_solution_bounds": finite_bound_count,
        }