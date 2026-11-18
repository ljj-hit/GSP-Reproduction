"""Tests for GSP-WA* priority, tree structure, experiences and return bounds."""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import Config
from environment import (
    ACTION_DELTAS,
    MazeEnvironment,
    Position,
    encode_state,
    path_to_actions,
    valid_action_mask,
)
from search import QGuidedWAStar, TransitionExperience


class ConstantActionQ(nn.Module):
    """A tiny deterministic stand-in for the Q network."""

    def __init__(self, values: Tuple[float, float, float, float]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.values.unsqueeze(0).expand(states.shape[0], -1)


class GoalDirectedQ(nn.Module):
    """Hand-coded negative-Manhattan Q values for open-grid tests."""

    def __init__(self, height: int = 8, width: int = 8) -> None:
        super().__init__()
        self.height = height
        self.width = width

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        batch = states.shape[0]
        flat = states.reshape(batch, -1)
        cell_count = self.height * self.width
        maze = flat[:, :cell_count].reshape(batch, self.height, self.width)
        agent = flat[:, cell_count:2 * cell_count].argmax(dim=1)
        goal = flat[:, 2 * cell_count:].argmax(dim=1)
        agent_row = torch.div(agent, self.width, rounding_mode="floor")
        agent_col = agent % self.width
        goal_row = torch.div(goal, self.width, rounding_mode="floor")
        goal_col = goal % self.width
        values = torch.empty((batch, 4), dtype=states.dtype, device=states.device)
        for action, (dr, dc) in enumerate(ACTION_DELTAS):
            next_row = agent_row + dr
            next_col = agent_col + dc
            inside = (
                (next_row >= 0)
                & (next_row < self.height)
                & (next_col >= 0)
                & (next_col < self.width)
            )
            safe_row = torch.clamp(next_row, 0, self.height - 1)
            safe_col = torch.clamp(next_col, 0, self.width - 1)
            free = maze[torch.arange(batch), safe_row, safe_col] == 0
            valid = inside & free
            distance = (next_row - goal_row).abs() + (next_col - goal_col).abs()
            values[:, action] = torch.where(
                valid,
                -distance.to(dtype=states.dtype),
                torch.full_like(distance, -1000, dtype=states.dtype),
            )
        return values


class CountingZeroQ(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: List[int] = []

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        batch = states.shape[0]
        flat = states.reshape(batch, -1)
        self.calls.extend(flat[:, 64:128].argmax(dim=1).tolist())
        return torch.zeros((batch, 4), dtype=states.dtype, device=states.device)


def _decode_position(state: np.ndarray, offset: int) -> Position:
    index = int(np.argmax(state[offset:offset + 64]))
    return index // 8, index % 8


def _edge_map(result) -> dict:
    mapping = {}
    for experience in result.experiences:
        key = (experience.action, int(np.argmax(experience.state[64:128])))
        mapping[key] = experience
    return mapping


def _assert_experience_is_legal(experience: TransitionExperience, maze: np.ndarray, goal: Position) -> None:
    position = _decode_position(experience.state, 64)
    next_position = _decode_position(experience.next_state, 64)
    assert _decode_position(experience.state, 128) == goal
    mask = valid_action_mask(maze, position)
    assert mask[experience.action], "Only legal actions may generate experiences"
    dr, dc = ACTION_DELTAS[experience.action]
    assert next_position == (position[0] + dr, position[1] + dc)
    np.testing.assert_allclose(
        experience.next_state, encode_state(maze, next_position, goal), atol=0
    )


def test_frontier_is_maximised() -> None:
    cfg = Config()
    maze = np.zeros((8, 8), dtype=np.int8)
    model = ConstantActionQ((0.0, 0.0, 0.0, 5.0))
    search = QGuidedWAStar(model, cfg, "cpu")
    result = search.search(maze, (0, 0), (0, 1))
    assert result.success
    # RIGHT has Q=5 and must be expanded before DOWN (Q=0).  If the heap
    # direction were minimised, DOWN would be expanded first.
    assert result.expansions == 1
    assert result.actions == [3]
    assert result.path == [(0, 0), (0, 1)]
    assert result.plan_length == 1


def test_solution_path_and_return_bounds_are_correct() -> None:
    cfg = Config()
    maze = np.zeros((8, 8), dtype=np.int8)
    start, goal = (1, 1), (5, 5)
    model = GoalDirectedQ()
    search = QGuidedWAStar(model, cfg, "cpu")
    result = search.search(maze, start, goal)
    assert result.success
    assert result.plan_length == 8
    assert result.path[0] == start and result.path[-1] == goal
    assert path_to_actions(result.path) == result.actions
    edges = _edge_map(result)
    for index, (position, action) in enumerate(zip(result.path[:-1], result.actions)):
        key = (action, position[0] * 8 + position[1])
        assert key in edges
        expected_return = -(result.plan_length - index)
        assert edges[key].return_bound == expected_return
        assert edges[key].dead_end is False
        assert edges[key].terminal == (index == result.plan_length - 1)
    assert sum(1 for exp in result.experiences if exp.terminal) == 1
    assert not any(exp.dead_end for exp in result.experiences)
    # The goal edge bootstraps no further Q values.
    terminal_experience = next(exp for exp in result.experiences if exp.terminal)
    np.testing.assert_allclose(
        terminal_experience.next_state,
        encode_state(maze, goal, goal),
    )


def test_ordinary_transitions_and_finite_path_bounds_are_distinguished() -> None:
    cfg = Config()
    maze = np.zeros((8, 8), dtype=np.int8)
    goal = (2, 2)
    model = ConstantActionQ((0.0, 0.0, 0.0, 0.0))
    result = QGuidedWAStar(model, cfg, "cpu").search(maze, (0, 0), goal)
    assert result.success
    assert len(result.experiences) == result.expansions
    finite = [exp for exp in result.experiences if math.isfinite(exp.return_bound)]
    ordinary = [exp for exp in result.experiences if math.isinf(exp.return_bound)]
    assert len(finite) == result.plan_length
    assert len(ordinary) > 0, "An open search should also expand off-path edges"
    assert all(exp.return_bound == -math.inf for exp in ordinary)
    assert all(not exp.dead_end for exp in result.experiences)
    for experience in result.experiences:
        _assert_experience_is_legal(experience, maze, goal)


def test_no_duplicate_edge_experience_and_first_discovery_tree() -> None:
    cfg = Config()
    maze = np.zeros((8, 8), dtype=np.int8)
    goal = (4, 4)
    model = CountingZeroQ()
    result = QGuidedWAStar(model, cfg, "cpu").search(maze, (0, 0), goal)
    assert result.success
    keys = [(exp.action, tuple(_decode_position(exp.state, 64))) for exp in result.experiences]
    assert len(keys) == len(set(keys))
    # Each newly discovered state is queried once; the goal does not need Q.
    assert len(model.calls) == len(set(model.calls))
    assert len(model.calls) == result.q_evaluations
    assert result.q_evaluations <= result.expansions + 1


def test_search_budget_and_failure_keep_ordinary_experiences() -> None:
    cfg = Config().copy_with(search_budget=5)
    maze = np.zeros((8, 8), dtype=np.int8)
    model = ConstantActionQ((0.0, 0.0, 0.0, 0.0))
    result = QGuidedWAStar(model, cfg, "cpu").search(maze, (7, 7), (0, 0))
    assert not result.success
    assert result.expansions == 5
    assert result.termination_reason == "expansion_budget"
    assert result.path == [] and result.actions == []
    assert len(result.experiences) == 5
    assert all(math.isinf(exp.return_bound) for exp in result.experiences)
    assert not any(exp.dead_end for exp in result.experiences)


def test_cul_de_sac_is_not_a_formal_dead_end() -> None:
    # A one-cell-wide pocket with walls on three sides still permits stepping
    # back, so A(s) is non-empty and it must not receive R_bottom.
    cfg = Config()
    maze = np.zeros((8, 8), dtype=np.int8)
    for row, col in ((2, 1), (1, 2), (2, 3), (3, 2)):
        maze[row, col] = 1
    goal = (5, 5)
    model = GoalDirectedQ()
    result = QGuidedWAStar(model, cfg, "cpu").search(maze, (0, 0), goal)
    assert result.success
    assert not any(exp.dead_end for exp in result.experiences)
    assert all(exp.return_bound != -64.0 for exp in result.experiences)


def test_path_positions_are_valid_four_neighbour_steps() -> None:
    cfg = Config()
    maze = np.zeros((8, 8), dtype=np.int8)
    goal = (6, 1)
    result = QGuidedWAStar(GoalDirectedQ(), cfg, "cpu").search(maze, (1, 6), goal)
    assert result.success
    environment = MazeEnvironment(maze, (1, 6), goal)
    for index, (position, action) in enumerate(zip(result.path[:-1], result.actions)):
        assert environment.get_successor(position, action) == result.path[index + 1]
    assert result.path[-1] == goal