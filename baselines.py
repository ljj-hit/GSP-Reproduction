"""Traditional search baselines and an optional greedy Q policy.

All searches use four-neighbour motion.  A*/WA* use first-discovery trees and
the same expansion budget as GSP-WA*.  BFS serves both as a shortest-path
oracle and as a baseline algorithm.
"""

from __future__ import annotations

import heapq
import time
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import Config
from environment import (
    MazeEnvironment,
    Position,
    encode_state,
    manhattan_distance,
    path_to_actions,
)
from search import SearchResult


def _reconstruct_path(
    start: Position,
    goal: Position,
    parent: dict[Position, Optional[Position]],
) -> List[Position]:
    path: List[Position] = []
    current: Optional[Position] = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    if not path or path[0] != start:
        raise RuntimeError("Broken search parent chain")
    return path


def run_bfs(
    maze: np.ndarray,
    start: Position,
    goal: Position,
    max_expansions: int = 128,
) -> SearchResult:
    """Breadth-first search with a node-expansion budget."""
    if max_expansions <= 0:
        raise ValueError("max_expansions must be positive")
    environment = MazeEnvironment(maze, start, goal)
    start, goal = environment.start, environment.goal
    if start == goal:
        return SearchResult(True, [start], [], 0, termination_reason="goal")

    frontier = deque([start])
    parent: dict[Position, Optional[Position]] = {start: None}
    expansions = 0
    success = False
    while frontier and expansions < max_expansions:
        position = frontier.popleft()
        expansions += 1
        if position == goal:
            success = True
            break
        for action in environment.get_valid_actions(position):
            nxt = environment.get_successor(position, action)
            if nxt not in parent:
                parent[nxt] = position
                frontier.append(nxt)

    if success:
        path = _reconstruct_path(start, goal, parent)
        return SearchResult(
            success=True,
            path=path,
            actions=path_to_actions(path),
            expansions=expansions,
            termination_reason="goal",
        )
    reason = "expansion_budget" if expansions >= max_expansions else "frontier_exhausted"
    return SearchResult(False, [], [], expansions, termination_reason=reason)


class HeuristicSearch:
    """A* (weight=1) or weighted A* (weight>1) with Manhattan distance."""

    def __init__(self, weight: float = 1.0, max_expansions: int = 128) -> None:
        if weight <= 0:
            raise ValueError("Heuristic weight must be positive")
        if max_expansions <= 0:
            raise ValueError("max_expansions must be positive")
        self.weight = float(weight)
        self.max_expansions = int(max_expansions)

    def search(
        self,
        maze: np.ndarray,
        start: Position,
        goal: Position,
    ) -> SearchResult:
        environment = MazeEnvironment(maze, start, goal)
        start, goal = environment.start, environment.goal
        if start == goal:
            return SearchResult(True, [start], [], 0, termination_reason="goal")

        counter = 0
        frontier: List[Tuple[float, int, Position]] = []
        parent: dict[Position, Optional[Position]] = {start: None}
        depth: dict[Position, int] = {start: 0}
        # priority = depth + weight * Manhattan; standard A* minimises it.
        start_priority = depth[start] + self.weight * manhattan_distance(start, goal)
        heapq.heappush(frontier, (start_priority, counter, start))
        counter += 1

        expansions = 0
        success = False
        while frontier and expansions < self.max_expansions:
            _, _, position = heapq.heappop(frontier)
            expansions += 1
            if position == goal:
                success = True
                break
            for action in environment.get_valid_actions(position):
                nxt = environment.get_successor(position, action)
                if nxt in parent:
                    continue
                parent[nxt] = position
                depth[nxt] = depth[position] + 1
                priority = depth[nxt] + self.weight * manhattan_distance(nxt, goal)
                heapq.heappush(frontier, (priority, counter, nxt))
                counter += 1

        if success:
            path = _reconstruct_path(start, goal, parent)
            return SearchResult(
                success=True,
                path=path,
                actions=path_to_actions(path),
                expansions=expansions,
                termination_reason="goal",
            )
        reason = "expansion_budget" if expansions >= self.max_expansions else "frontier_exhausted"
        return SearchResult(False, [], [], expansions, termination_reason=reason)


def run_astar(
    maze: np.ndarray,
    start: Position,
    goal: Position,
    max_expansions: int = 128,
) -> SearchResult:
    return HeuristicSearch(weight=1.0, max_expansions=max_expansions).search(maze, start, goal)


def run_weighted_astar(
    maze: np.ndarray,
    start: Position,
    goal: Position,
    weight: float = 2.0,
    max_expansions: int = 128,
) -> SearchResult:
    return HeuristicSearch(weight=weight, max_expansions=max_expansions).search(maze, start, goal)


class GreedyQPolicy:
    """Direct ``argmax_a Q(s,a)`` rollout with cycle detection.

    This is an optional diagnostic for the learned Q function, not the main
    GSP-WA* evaluator.  It is capped by an explicit step limit.
    """

    def __init__(
        self,
        network: nn.Module,
        config: Config,
        device: str | torch.device,
        *,
        max_steps: Optional[int] = None,
    ) -> None:
        self.network = network
        self.config = config
        self.device = torch.device(device)
        self.max_steps = int(config.greedy_max_steps if max_steps is None else max_steps)
        if self.max_steps <= 0:
            raise ValueError("Greedy max_steps must be positive")
        for parameter in network.parameters():
            if parameter.device != self.device:
                raise ValueError(
                    f"Greedy policy network is not on requested device {self.device}"
                )

    @torch.no_grad()
    def _q(self, maze: np.ndarray, position: Position, goal: Position) -> np.ndarray:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        encoded = encode_state(maze, position, goal)
        tensor = torch.from_numpy(encoded).to(self.device, dtype=torch.float32).unsqueeze(0)
        q_values = self.network(tensor).detach().cpu().numpy().reshape(-1)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._inference_time += time.perf_counter() - start
        if q_values.shape[0] != 4:
            raise RuntimeError("Q network must output four action values")
        if not np.all(np.isfinite(q_values)):
            raise FloatingPointError("Greedy Q policy encountered non-finite Q values")
        return q_values

    def rollout(self, maze: np.ndarray, start: Position, goal: Position) -> SearchResult:
        environment = MazeEnvironment(maze, start, goal)
        start, goal = environment.start, environment.goal
        was_training = bool(getattr(self.network, "training", False))
        old_flags = [bool(p.requires_grad) for p in self.network.parameters()]
        self.network.eval()
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        self._inference_time = 0.0
        q_evaluations = 0
        try:
            position = start
            positions: List[Position] = [position]
            actions: List[int] = []
            visited = {position}
            termination_reason = "step_limit"
            success = False
            decisions = 0
            while decisions < self.max_steps:
                if position == goal:
                    success = True
                    termination_reason = "goal"
                    break
                valid_actions = environment.get_valid_actions(position)
                if not valid_actions:
                    termination_reason = "no_actions"
                    break
                q_values = self._q(environment.maze, position, goal)
                q_evaluations += 1
                # Tie-break toward the smallest action id for determinism.
                action = max(valid_actions, key=lambda a: (float(q_values[a]), -a))
                decisions += 1
                next_position = environment.get_successor(position, action)
                if next_position in visited:
                    termination_reason = "cycle"
                    break
                visited.add(next_position)
                actions.append(action)
                positions.append(next_position)
                position = next_position
                if position == goal:
                    success = True
                    termination_reason = "goal"
                    break
            return SearchResult(
                success=success,
                path=positions,
                actions=actions,
                expansions=decisions,
                experiences=[],
                termination_reason=termination_reason,
                inference_time_seconds=self._inference_time,
                q_evaluations=q_evaluations,
            )
        finally:
            for parameter, flag in zip(self.network.parameters(), old_flags):
                parameter.requires_grad_(flag)
            self.network.train(was_training)


# Public aliases with descriptive names used by evaluation/README.
def bfs_search(maze: np.ndarray, start: Position, goal: Position, max_expansions: int = 128) -> SearchResult:
    return run_bfs(maze, start, goal, max_expansions)


def astar_search(maze: np.ndarray, start: Position, goal: Position, max_expansions: int = 128) -> SearchResult:
    return run_astar(maze, start, goal, max_expansions)


def weighted_astar_search(
    maze: np.ndarray,
    start: Position,
    goal: Position,
    weight: float = 2.0,
    max_expansions: int = 128,
) -> SearchResult:
    return run_weighted_astar(maze, start, goal, weight, max_expansions)