"""Unit tests for maze transitions, legal actions and state encoding."""

from __future__ import annotations

import numpy as np
import pytest

from environment import (
    ACTION_DELTAS,
    DOWN,
    LEFT,
    NUM_ACTIONS,
    RIGHT,
    UP,
    InvalidActionError,
    InvalidMazeError,
    MazeEnvironment,
    bfs_shortest_path,
    bfs_shortest_path_length,
    encode_state,
    one_hot_position,
    path_to_actions,
)


def open_maze() -> np.ndarray:
    return np.zeros((8, 8), dtype=np.int8)


def test_action_ids_and_deltas_are_fixed() -> None:
    assert (UP, DOWN, LEFT, RIGHT) == (0, 1, 2, 3)
    assert ACTION_DELTAS == ((-1, 0), (1, 0), (0, -1), (0, 1))
    assert NUM_ACTIONS == 4


def test_valid_actions_respect_boundaries_and_walls() -> None:
    maze = open_maze()
    maze[2, 3] = 1
    env = MazeEnvironment(maze, start=(2, 2), goal=(7, 7))
    actions = env.get_valid_actions((2, 2))
    assert RIGHT not in actions, "RIGHT enters a wall and must be illegal"
    assert actions == [UP, DOWN, LEFT]

    assert env.get_valid_actions((0, 0)) == [DOWN, RIGHT]
    assert env.get_valid_actions((7, 7)) == [UP, LEFT]


def test_isolated_free_cell_has_no_legal_actions() -> None:
    maze = open_maze()
    # Enclose (3, 3) with walls but leave it itself free.  This cell is
    # unreachable from start, yet A(s)=empty is the formal dead-end condition.
    for dr, dc in ACTION_DELTAS:
        maze[3 + dr, 3 + dc] = 1
    assert MazeEnvironment(maze, start=(0, 0), goal=(7, 7)).get_valid_actions((3, 3)) == []


def test_legal_step_has_reward_minus_one_and_updates_position() -> None:
    env = MazeEnvironment(open_maze(), start=(0, 0), goal=(0, 2))
    result = env.step(RIGHT)
    assert result.position == (0, 1)
    assert result.reward == -1.0
    assert result.done is False
    assert env.agent_position == (0, 1)
    assert env.steps == 1


def test_illegal_action_raises_without_state_change_or_reward() -> None:
    maze = open_maze()
    maze[0, 1] = 1
    env = MazeEnvironment(maze, start=(0, 0), goal=(7, 7))
    with pytest.raises(InvalidActionError):
        env.step(RIGHT)
    assert env.agent_position == (0, 0)
    assert env.steps == 0
    assert not env.done

    with pytest.raises(InvalidActionError):
        env.step(UP)  # out of bounds
    assert env.agent_position == (0, 0)

    with pytest.raises(InvalidActionError):
        env.step(99)
    assert env.steps == 0


def test_goal_terminates_and_no_further_step_is_allowed() -> None:
    env = MazeEnvironment(open_maze(), start=(0, 0), goal=(0, 2))
    env.step(RIGHT)
    result = env.step(RIGHT)
    assert result.done is True
    assert env.done is True
    with pytest.raises(RuntimeError):
        env.step(DOWN)
    assert env.reset() == (0, 0)
    assert env.steps == 0
    assert not env.done


def test_invalid_start_or_goal_is_rejected() -> None:
    maze = open_maze()
    maze[0, 0] = 1
    with pytest.raises(ValueError):
        MazeEnvironment(maze, start=(0, 0), goal=(1, 1))
    with pytest.raises(ValueError):
        MazeEnvironment(maze, start=(1, 1), goal=(1, 1))
    with pytest.raises(ValueError):
        MazeEnvironment(open_maze(), start=(0, 0), goal=(0, 0))


def test_maze_validation_rejects_non_binary_arrays() -> None:
    with pytest.raises(InvalidMazeError):
        MazeEnvironment(np.full((8, 8), 2), start=(0, 0), goal=(1, 1))
    with pytest.raises(InvalidMazeError):
        MazeEnvironment(np.zeros((8,), dtype=int), start=(0, 0), goal=(1, 1))


def test_state_encoding_dimension_channels_and_one_hot() -> None:
    maze = open_maze()
    maze[1, 2] = 1
    position = (3, 4)
    goal = (7, 6)
    encoded = encode_state(maze, position, goal)
    assert encoded.shape == (192,)
    assert encoded.dtype == np.float32
    assert np.all((encoded == 0.0) | (encoded == 1.0))
    np.testing.assert_array_equal(encoded[:64], maze.reshape(-1).astype(np.float32))
    np.testing.assert_array_equal(
        encoded[64:128], one_hot_position(position, 8, 8)
    )
    np.testing.assert_array_equal(encoded[128:192], one_hot_position(goal, 8, 8))
    assert encoded[64:128].sum() == 1.0
    assert encoded[128:192].sum() == 1.0
    assert encoded[64 + position[0] * 8 + position[1]] == 1.0
    assert encoded[128 + goal[0] * 8 + goal[1]] == 1.0


def test_state_encoding_rejects_impossible_positions() -> None:
    maze = open_maze()
    maze[2, 2] = 1
    with pytest.raises(ValueError):
        encode_state(maze, (2, 2), (7, 7))
    with pytest.raises(ValueError):
        encode_state(maze, (0, 0), (2, 2))
    with pytest.raises(ValueError):
        encode_state(maze, (8, 0), (7, 7))


def test_bfs_shortest_path_and_length() -> None:
    maze = open_maze()
    path = bfs_shortest_path(maze, (0, 0), (0, 3))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (0, 3)
    assert len(path) - 1 == 3 == bfs_shortest_path_length(maze, (0, 0), (0, 3))
    assert path_to_actions(path) == [RIGHT, RIGHT, RIGHT]


def test_bfs_detects_unreachable_goal() -> None:
    maze = open_maze()
    maze[:, 4] = 1
    maze[0, 4] = 0
    maze[7, 4] = 0
    # Now the barrier has openings, so it is reachable.
    assert bfs_shortest_path(maze, (0, 0), (0, 7)) is not None

    maze[0, 4] = 1
    maze[7, 4] = 1
    assert bfs_shortest_path(maze, (0, 0), (0, 7)) is None


def test_path_to_actions_rejects_diagonal_steps() -> None:
    with pytest.raises(ValueError):
        path_to_actions([(0, 0), (1, 1)])