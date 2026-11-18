"""Grid-world maze environment and state encoding.

The world is a fixed-size 2-D grid.  Walls occupy cells rather than edges.
The action numbering is shared by every module:

    UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

Position = Tuple[int, int]

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTION_DELTAS: Tuple[Position, ...] = (
    (-1, 0),  # UP
    (1, 0),   # DOWN
    (0, -1),  # LEFT
    (0, 1),   # RIGHT
)
ACTION_NAMES: Tuple[str, ...] = ("UP", "DOWN", "LEFT", "RIGHT")
NUM_ACTIONS = 4


class InvalidActionError(ValueError):
    """Raised when an illegal move is passed to the environment."""


class InvalidMazeError(ValueError):
    """Raised for malformed maze data."""


@dataclass
class MazeInstance:
    """A maze plus its start/goal and optional generation metadata."""

    maze: np.ndarray
    start: Position
    goal: Position
    instance_id: int = 0
    bfs_length: Optional[int] = None
    difficulty: str = "unknown"
    seed: Optional[int] = None
    map_hash: str = ""


@dataclass
class StepResult:
    """Result of a legal environment transition."""

    position: Position
    reward: float
    done: bool


def validate_maze(maze: np.ndarray) -> np.ndarray:
    """Validate a binary obstacle matrix and return it as an int8 array."""
    array = np.asarray(maze)
    if array.ndim != 2:
        raise InvalidMazeError("Maze must be a two-dimensional array")
    if array.size == 0:
        raise InvalidMazeError("Maze must not be empty")
    if not np.issubdtype(array.dtype, np.integer) and not np.issubdtype(array.dtype, np.floating):
        raise InvalidMazeError("Maze entries must be numeric 0/1 values")
    if not np.all(np.isin(array, (0, 1))):
        raise InvalidMazeError("Maze entries must be 0 (free) or 1 (obstacle)")
    return array.astype(np.int8, copy=False)


def in_bounds(maze: np.ndarray, position: Position) -> bool:
    row, col = position
    return 0 <= row < maze.shape[0] and 0 <= col < maze.shape[1]


def is_free(maze: np.ndarray, position: Position) -> bool:
    return in_bounds(maze, position) and int(maze[position[0], position[1]]) == 0


def valid_actions_for(maze: np.ndarray, position: Position) -> List[int]:
    """Return all legal actions at ``position`` in ascending action order.

    A move is legal only if it stays inside the map and the destination cell
    is free.  Actions into walls are excluded rather than treated as no-ops.
    """
    maze = validate_maze(maze)
    if not in_bounds(maze, position):
        raise ValueError(f"Position out of bounds: {position}")
    if not is_free(maze, position):
        raise ValueError(f"Position is an obstacle: {position}")
    actions: List[int] = []
    row, col = position
    for action, (dr, dc) in enumerate(ACTION_DELTAS):
        nr, nc = row + dr, col + dc
        if 0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1] and maze[nr, nc] == 0:
            actions.append(action)
    return actions


def valid_action_mask(maze: np.ndarray, position: Position) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    for action in valid_actions_for(maze, position):
        mask[action] = True
    return mask


def successor_position(position: Position, action: int) -> Position:
    """Return the formal successor; legality must be checked separately."""
    if action not in (UP, DOWN, LEFT, RIGHT):
        raise InvalidActionError(f"Unknown action id: {action}")
    dr, dc = ACTION_DELTAS[action]
    return position[0] + dr, position[1] + dc


def encode_state(
    maze: np.ndarray,
    agent_position: Position,
    goal: Position,
) -> np.ndarray:
    """Encode ``(M, p, g)`` as ``[M_flat, onehot(p), onehot(g)]``.

    The output is a float32 vector with three row-major channels.  For the
    required 8 x 8 grid it has exactly 192 entries.
    """
    maze = validate_maze(maze)
    height, width = maze.shape
    if not in_bounds(maze, agent_position):
        raise ValueError(f"Agent position out of bounds: {agent_position}")
    if not in_bounds(maze, goal):
        raise ValueError(f"Goal position out of bounds: {goal}")
    if not is_free(maze, goal):
        raise ValueError(f"Goal position is an obstacle: {goal}")
    if not is_free(maze, agent_position):
        raise ValueError(f"Agent position is an obstacle: {agent_position}")

    obstacle_channel = maze.reshape(-1).astype(np.float32)
    agent_channel = np.zeros(height * width, dtype=np.float32)
    goal_channel = np.zeros(height * width, dtype=np.float32)
    agent_channel[agent_position[0] * width + agent_position[1]] = 1.0
    goal_channel[goal[0] * width + goal[1]] = 1.0
    encoded = np.concatenate((obstacle_channel, agent_channel, goal_channel), axis=0)
    return encoded.astype(np.float32, copy=False)


def one_hot_position(position: Position, height: int, width: int) -> np.ndarray:
    """Utility used by tests/diagnostics to expose the position encoding."""
    row, col = position
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError("Position out of bounds")
    encoded = np.zeros(height * width, dtype=np.float32)
    encoded[row * width + col] = 1.0
    return encoded


def maze_map_hash(maze: np.ndarray) -> str:
    """Stable SHA-256 hash of the obstacle layout."""
    array = validate_maze(maze).astype(np.uint8, copy=False)
    header = f"{array.shape[0]}x{array.shape[1]}|uint8|".encode("ascii")
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def bfs_shortest_path(
    maze: np.ndarray,
    start: Position,
    goal: Position,
    max_expansions: Optional[int] = None,
) -> Optional[List[Position]]:
    """Return a shortest four-neighbour path, or ``None`` if unreachable.

    This pure reference implementation is used by the generator and for path
    ratio metrics.  It is never used to generate Q-learning labels.
    """
    maze = validate_maze(maze)
    if not is_free(maze, start):
        raise ValueError("Start must be inside the maze and free")
    if not is_free(maze, goal):
        raise ValueError("Goal must be inside the maze and free")
    if start == goal:
        return [start]

    frontier: deque[Position] = deque([start])
    parent: dict[Position, Optional[Position]] = {start: None}
    expansions = 0
    found = False
    while frontier:
        if max_expansions is not None and expansions >= max_expansions:
            break
        position = frontier.popleft()
        expansions += 1
        if position == goal:
            found = True
            break
        row, col = position
        for dr, dc in ACTION_DELTAS:
            nxt = (row + dr, col + dc)
            if not in_bounds(maze, nxt) or not is_free(maze, nxt):
                continue
            if nxt not in parent:
                parent[nxt] = position
                frontier.append(nxt)

    if not found:
        return None
    path: List[Position] = []
    current: Optional[Position] = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def bfs_shortest_path_length(
    maze: np.ndarray,
    start: Position,
    goal: Position,
) -> Optional[int]:
    path = bfs_shortest_path(maze, start, goal)
    return None if path is None else len(path) - 1


def manhattan_distance(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class MazeEnvironment:
    """Stateful interface for a single maze instance.

    Search uses the pure helpers above so that it can evaluate successor
    states without mutating this object.  The stateful API (``step``) is used
    for replay-style rollouts and tests.
    """

    def __init__(
        self,
        maze: np.ndarray,
        start: Position,
        goal: Position,
        *,
        copy_maze: bool = True,
    ) -> None:
        array = validate_maze(maze)
        self.maze = array.copy() if copy_maze else array
        self.start = (int(start[0]), int(start[1]))
        self.goal = (int(goal[0]), int(goal[1]))
        if not in_bounds(self.maze, self.start):
            raise ValueError(f"Start out of bounds: {self.start}")
        if not in_bounds(self.maze, self.goal):
            raise ValueError(f"Goal out of bounds: {self.goal}")
        if not is_free(self.maze, self.start):
            raise ValueError("Start cannot be an obstacle")
        if not is_free(self.maze, self.goal):
            raise ValueError("Goal cannot be an obstacle")
        if self.start == self.goal:
            raise ValueError("Start and goal must be different cells")
        self.agent_position: Position = self.start
        self.done: bool = False
        self.steps: int = 0

    @property
    def height(self) -> int:
        return int(self.maze.shape[0])

    @property
    def width(self) -> int:
        return int(self.maze.shape[1])

    @property
    def state_dim(self) -> int:
        return 3 * self.height * self.width

    def reset(self) -> Position:
        self.agent_position = self.start
        self.done = False
        self.steps = 0
        return self.agent_position

    def get_valid_actions(self, position: Optional[Position] = None) -> List[int]:
        target = self.agent_position if position is None else position
        return valid_actions_for(self.maze, target)

    def get_valid_action_mask(self, position: Optional[Position] = None) -> np.ndarray:
        target = self.agent_position if position is None else position
        return valid_action_mask(self.maze, target)

    def get_successor(self, position: Position, action: int) -> Position:
        """Return a legal successor or raise :class:`InvalidActionError`."""
        if not in_bounds(self.maze, position):
            raise InvalidActionError(f"Position out of bounds: {position}")
        if not is_free(self.maze, position):
            raise InvalidActionError(f"Position is an obstacle: {position}")
        if action not in (UP, DOWN, LEFT, RIGHT):
            raise InvalidActionError(f"Unknown action id: {action}")
        nxt = successor_position(position, action)
        if not in_bounds(self.maze, nxt):
            raise InvalidActionError(
                f"Action {ACTION_NAMES[action]} from {position} leaves the maze"
            )
        if not is_free(self.maze, nxt):
            raise InvalidActionError(
                f"Action {ACTION_NAMES[action]} from {position} enters an obstacle"
            )
        return nxt

    def step(self, action: int) -> StepResult:
        """Apply one legal action.

        Illegal actions raise instead of self-looping.  Each real movement has
        reward -1 and reaching the goal terminates the episode.
        """
        if self.done:
            raise RuntimeError("Cannot step a finished episode; call reset() first")
        if action not in (UP, DOWN, LEFT, RIGHT):
            raise InvalidActionError(f"Unknown action id: {action}")
        next_position = self.get_successor(self.agent_position, action)
        self.agent_position = next_position
        self.steps += 1
        self.done = self.agent_position == self.goal
        return StepResult(position=next_position, reward=-1.0, done=self.done)

    def encode_state(self, position: Optional[Position] = None) -> np.ndarray:
        target = self.agent_position if position is None else position
        return encode_state(self.maze, target, self.goal)

    def render(self) -> str:
        from utils import maze_to_text

        return maze_to_text(self.maze, self.start, self.goal, [self.agent_position])

    @classmethod
    def from_instance(cls, instance: MazeInstance) -> "MazeEnvironment":
        return cls(instance.maze, instance.start, instance.goal)

    def __repr__(self) -> str:
        return (
            f"MazeEnvironment(start={self.start}, goal={self.goal}, "
            f"position={self.agent_position}, done={self.done})"
        )


def positions_from_actions(start: Position, actions: Sequence[int]) -> List[Position]:
    """Convert an action sequence into positions (legality not checked)."""
    positions = [start]
    current = start
    for action in actions:
        current = successor_position(current, int(action))
        positions.append(current)
    return positions


def path_to_actions(path: Sequence[Position]) -> List[int]:
    """Convert a contiguous position path into action ids."""
    actions: List[int] = []
    for (r0, c0), (r1, c1) in zip(path[:-1], path[1:]):
        delta = (r1 - r0, c1 - c0)
        if delta not in ACTION_DELTAS:
            raise ValueError(f"Path contains a non-cardinal step: {delta}")
        actions.append(ACTION_DELTAS.index(delta))
    return actions