"""GSP's Q-guided Weighted A* search.

Nodes are state-action pairs.  With a fixed maze and goal, the position alone
identifies a search state.  The implementation follows a first-discovery
search tree: a state is expanded once, parents are never rewired, and each
new state's four Q values are computed exactly once and cached.

Priority (maximised):

    f(s, a) = g(s) + w * Q_theta(s, a),   g(s) = -depth(s)

Successful searches produce finite lower-bound returns along the solution
path; ordinary transitions keep ``return_bound = -inf``.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import Config
from environment import (
    ACTION_NAMES,
    MazeEnvironment,
    MazeInstance,
    Position,
    encode_state,
    path_to_actions,
    valid_action_mask,
)

EdgeKey = Tuple[Position, int]


@dataclass
class TransitionExperience:
    """One state-action transition collected by a search."""

    state: np.ndarray
    action: int
    next_state: np.ndarray
    return_bound: float
    terminal: bool
    dead_end: bool
    next_valid_mask: np.ndarray

    def __post_init__(self) -> None:
        self.state = np.asarray(self.state, dtype=np.float32)
        self.next_state = np.asarray(self.next_state, dtype=np.float32)
        self.next_valid_mask = np.asarray(self.next_valid_mask, dtype=np.bool_)
        self.action = int(self.action)
        self.return_bound = float(self.return_bound)
        self.terminal = bool(self.terminal)
        self.dead_end = bool(self.dead_end)
        if self.state.ndim != 1 or self.next_state.ndim != 1:
            raise ValueError("Experience states must be one-dimensional")
        if self.state.shape != self.next_state.shape:
            raise ValueError("State and next_state must have the same shape")
        if not 0 <= self.action < 4:
            raise ValueError("Action must be in [0, 3]")
        if self.next_valid_mask.shape != (4,):
            raise ValueError("next_valid_mask must have shape (4,)")


@dataclass
class SearchResult:
    """Common result type for GSP-WA*, baselines and greedy rollouts."""

    success: bool
    path: List[Position] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    expansions: int = 0
    experiences: List[TransitionExperience] = field(default_factory=list)
    termination_reason: str = "unknown"
    inference_time_seconds: float = 0.0
    q_evaluations: int = 0
    dead_end_transitions: int = 0

    @property
    def plan_length(self) -> int:
        if self.actions:
            return len(self.actions)
        if len(self.path) > 1:
            return len(self.path) - 1
        return 0

    @property
    def frontier_empty(self) -> bool:
        return self.termination_reason == "frontier_exhausted"

    @property
    def hit_budget(self) -> bool:
        return self.termination_reason == "expansion_budget"

    @classmethod
    def failure(cls, reason: str, expansions: int) -> "SearchResult":
        return cls(
            success=False,
            path=[],
            actions=[],
            expansions=int(expansions),
            experiences=[],
            termination_reason=reason,
        )


def _positions_from_actions(start: Position, actions: Sequence[int]) -> List[Position]:
    from environment import positions_from_actions

    return positions_from_actions(start, actions)


class QGuidedWAStar:
    """Q-network-guided Weighted A* with state-action frontier nodes."""

    def __init__(
        self,
        network: nn.Module,
        config: Config,
        device: str | torch.device,
        *,
        weight: Optional[float] = None,
        search_budget: Optional[int] = None,
    ) -> None:
        self.network = network
        self.config = config
        self.device = torch.device(device)
        self.weight = float(config.wa_star_weight if weight is None else weight)
        self.search_budget = int(config.search_budget if search_budget is None else search_budget)
        if self.weight <= 0:
            raise ValueError("WA* weight must be positive")
        if self.search_budget <= 0:
            raise ValueError("Search budget must be positive")
        if not self._network_and_device_aligned():
            raise ValueError(
                f"Network parameters are not on requested search device {self.device}"
            )

    def _network_and_device_aligned(self) -> bool:
        for parameter in self.network.parameters():
            if parameter.device != self.device:
                return False
        for buffer in self.network.buffers():
            if buffer.device != self.device:
                return False
        return True

    def _frozen_network(self) -> Tuple[bool, List[bool]]:
        was_training = bool(getattr(self.network, "training", False))
        old_flags = [bool(p.requires_grad) for p in self.network.parameters()]
        self.network.eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        return was_training, old_flags

    def _unfreeze_network(self, was_training: bool, old_flags: Sequence[bool]) -> None:
        for parameter, flag in zip(self.network.parameters(), old_flags):
            parameter.requires_grad_(flag)
        self.network.train(was_training)

    def _evaluate_q(self, maze: np.ndarray, position: Position, goal: Position) -> np.ndarray:
        """Batch-compute all four action values for one state."""
        start_time = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        try:
            encoded = encode_state(maze, position, goal)
            tensor = torch.from_numpy(encoded).to(self.device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.network(tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            q_values = q_values.detach().to("cpu").numpy().reshape(-1)
        except Exception:
            # Ensure an exception does not leave timing accounting ambiguous.
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            raise
        self._inference_time += time.perf_counter() - start_time
        self._q_evaluations += 1
        if q_values.shape[0] != 4:
            raise RuntimeError(f"Q network must output 4 values, got {q_values.shape[0]}")
        if not np.all(np.isfinite(q_values)):
            raise FloatingPointError(f"Non-finite Q values at position {position}: {q_values}")
        q_values = q_values.astype(np.float64, copy=False)
        self._q_cache[position] = q_values
        return q_values

    def _q(self, maze: np.ndarray, position: Position, goal: Position) -> np.ndarray:
        cached = self._q_cache.get(position)
        if cached is not None:
            return cached
        return self._evaluate_q(maze, position, goal)

    def search(
        self,
        maze: np.ndarray,
        start: Position,
        goal: Position,
    ) -> SearchResult:
        """Run one search episode on a single maze."""
        environment = MazeEnvironment(maze, start, goal)
        start = environment.start
        goal = environment.goal

        self._q_cache: Dict[Position, np.ndarray] = {}
        self._inference_time = 0.0
        self._q_evaluations = 0
        self._counter = 0

        was_training, old_flags = self._frozen_network()
        try:
            # Frontier is a max-heap.  (-priority, unique sequence, position, action)
            frontier: List[Tuple[float, int, Position, int]] = []
            visited: set[Position] = {start}
            depth: Dict[Position, int] = {start: 0}
            parent: Dict[Position, Optional[Tuple[Position, int]]] = {start: None}
            transitions: Dict[EdgeKey, TransitionExperience] = {}

            start_q = self._q(environment.maze, start, goal)
            for action in environment.get_valid_actions(start):
                priority = -depth[start] + self.weight * float(start_q[action])
                heapq.heappush(
                    frontier,
                    (-priority, self._counter, start, action),
                )
                self._counter += 1

            expansions = 0
            success = False
            termination_reason = "frontier_exhausted"

            while frontier and expansions < self.search_budget:
                negative_priority, _, position, action = heapq.heappop(frontier)
                del negative_priority
                expansions += 1
                depth_here = depth[position]
                next_position = environment.get_successor(position, action)
                states_encoded_current = encode_state(environment.maze, position, goal)
                states_encoded_next = encode_state(environment.maze, next_position, goal)
                next_mask = valid_action_mask(environment.maze, next_position)

                if next_position == goal:
                    transition = TransitionExperience(
                        state=states_encoded_current,
                        action=action,
                        next_state=states_encoded_next,
                        return_bound=-1.0,
                        terminal=True,
                        dead_end=False,
                        next_valid_mask=next_mask,
                    )
                    transitions[(position, action)] = transition
                    parent[goal] = (position, action)
                    success = True
                    termination_reason = "goal"
                    break

                valid_next_actions = environment.get_valid_actions(next_position)
                if len(valid_next_actions) == 0:
                    # A true dead end, as defined by A(s') = empty.  This is not
                    # the same as a visual cul-de-sac: reverse edges remain legal.
                    transition = TransitionExperience(
                        state=states_encoded_current,
                        action=action,
                        next_state=states_encoded_next,
                        return_bound=float(self.config.dead_end_return),
                        terminal=False,
                        dead_end=True,
                        next_valid_mask=next_mask,
                    )
                    transitions[(position, action)] = transition
                    continue

                transition = TransitionExperience(
                    state=states_encoded_current,
                    action=action,
                    next_state=states_encoded_next,
                    return_bound=-math.inf,
                    terminal=False,
                    dead_end=False,
                    next_valid_mask=next_mask,
                )
                transitions[(position, action)] = transition

                if next_position not in visited:
                    visited.add(next_position)
                    depth[next_position] = depth_here + 1
                    parent[next_position] = (position, action)
                    next_q = self._q(environment.maze, next_position, goal)
                    next_g = -depth[next_position]
                    for next_action in valid_next_actions:
                        priority = next_g + self.weight * float(next_q[next_action])
                        heapq.heappush(
                            frontier,
                            (-priority, self._counter, next_position, next_action),
                        )
                        self._counter += 1

            if not success:
                termination_reason = (
                    "expansion_budget"
                    if expansions >= self.search_budget
                    else "frontier_exhausted"
                )

            actions_to_goal: List[int] = []
            path: List[Position] = []
            if success:
                # Reconstruct the first-discovery path from goal to start.
                temp_actions: List[int] = []
                current = goal
                while parent[current] is not None:
                    previous, edge_action = parent[current]  # type: ignore[misc]
                    temp_actions.append(edge_action)
                    current = previous
                actions_to_goal = list(reversed(temp_actions))
                path = _positions_from_actions(start, actions_to_goal)

                # Assign finite search lower bounds: final action -1, then -2, ...
                return_value = 0.0
                current = goal
                while parent[current] is not None:
                    previous, edge_action = parent[current]  # type: ignore[misc]
                    return_value -= 1.0
                    edge = transitions.get((previous, edge_action))
                    if edge is None:
                        raise RuntimeError("Solution path edge is missing from search tree")
                    edge.return_bound = return_value
                    current = previous

            experiences = list(transitions.values())
            dead_end_count = sum(1 for experience in experiences if experience.dead_end)
            return SearchResult(
                success=success,
                path=path,
                actions=actions_to_goal,
                expansions=expansions,
                experiences=experiences,
                termination_reason=termination_reason,
                inference_time_seconds=self._inference_time,
                q_evaluations=self._q_evaluations,
                dead_end_transitions=dead_end_count,
            )
        finally:
            self._unfreeze_network(was_training, old_flags)

    def search_instance(self, instance: MazeInstance) -> SearchResult:
        return self.search(instance.maze, instance.start, instance.goal)


def actions_from_path(path: Sequence[Position]) -> List[int]:
    """Public helper shared with evaluation code."""
    return path_to_actions(path)


def describe_action(action: int) -> str:
    if action not in range(len(ACTION_NAMES)):
        raise ValueError(f"Unknown action {action}")
    return ACTION_NAMES[action]