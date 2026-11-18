"""Procedural generation of maze datasets.

The generator creates three disjoint splits with distinct master seeds:
1000 training, 200 validation and 300 test mazes under the default config.
Each accepted maze stores its obstacle matrix, start, goal, per-maze seed,
BFS shortest-path length, difficulty label and a layout hash.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from config import Config
from environment import (
    MazeInstance,
    bfs_shortest_path,
    is_free,
    maze_map_hash,
    validate_maze,
)
from utils import ensure_dir, save_json, setup_logger, utc_now_iso


DIFFICULTIES: Tuple[str, str, str] = ("easy", "medium", "hard")


@dataclass
class GenerationStats:
    attempts: int = 0
    accepted: int = 0
    unreachable: int = 0
    too_short: int = 0
    duplicate: int = 0
    wrong_difficulty: int = 0
    malformed: int = 0


def difficulty_for_length(length: int, config: Config) -> Optional[str]:
    """Map BFS path length to the project's three difficulty bands."""
    if length < config.min_bfs_length:
        return None
    if length <= config.easy_max_bfs_length:
        return "easy"
    if length <= config.medium_max_bfs_length:
        return "medium"
    return "hard"


def _difficulty_quotas(count: int) -> Dict[str, int]:
    """Split ``count`` into balanced, deterministic target quotas."""
    if count <= 0:
        return {name: 0 for name in DIFFICULTIES}
    base = count // 3
    remainder = count % 3
    quotas = {
        "easy": base + (1 if remainder >= 1 else 0),
        "medium": base + (1 if remainder >= 2 else 0),
        "hard": base,
    }
    return quotas


def _sample_candidate(config: Config, rng: np.random.Generator) -> Optional[Dict[str, Any]]:
    """Generate one reachable candidate maze.  Reject unreachable/too-short cases."""
    height = config.height
    width = config.width
    cell_count = height * width
    maze_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
    local_rng = np.random.default_rng(maze_seed)

    ratio = float(local_rng.uniform(config.min_obstacle_ratio, config.max_obstacle_ratio))
    min_obstacles = int(math.ceil(config.min_obstacle_ratio * cell_count))
    max_obstacles = int(math.floor(config.max_obstacle_ratio * cell_count))
    n_obstacles = int(round(ratio * cell_count))
    n_obstacles = max(min_obstacles, min(max_obstacles, n_obstacles))

    maze = np.zeros((height, width), dtype=np.int8)
    obstacle_flat = local_rng.choice(cell_count, size=n_obstacles, replace=False)
    maze.reshape(-1)[obstacle_flat] = 1
    free_flat = np.flatnonzero(maze.reshape(-1) == 0)
    if free_flat.size < 2:
        return None
    start_idx, goal_idx = local_rng.choice(free_flat, size=2, replace=False)
    start = (int(start_idx // width), int(start_idx % width))
    goal = (int(goal_idx // width), int(goal_idx % width))

    path = bfs_shortest_path(maze, start, goal)
    if path is None:
        return None
    path_length = len(path) - 1
    difficulty = difficulty_for_length(path_length, config)
    if difficulty is None:
        return None

    return {
        "maze": maze,
        "start": start,
        "goal": goal,
        "seed": maze_seed,
        "bfs_length": path_length,
        "difficulty": difficulty,
        "obstacle_ratio": float(n_obstacles / cell_count),
        "map_hash": maze_map_hash(maze),
    }


def _record_category(record: Dict[str, Any]) -> str:
    return str(record["difficulty"])


def _generate_split(
    config: Config,
    split_name: str,
    count: int,
    seed: int,
    used_map_hashes: Set[str],
    logger: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate one split with difficulty quotas and bounded attempts."""
    rng = np.random.default_rng(seed)
    stats = GenerationStats()
    records: List[Dict[str, Any]] = []
    accepted_hashes: Set[str] = set()
    per_target: Dict[str, Dict[str, int]] = {
        name: {"requested": 0, "produced": 0, "attempts": 0} for name in DIFFICULTIES
    }
    quotas = _difficulty_quotas(count)

    def try_accept(record: Optional[Dict[str, Any]], target: Optional[str]) -> bool:
        if record is None:
            stats.unreachable += 1
            return False
        if record["map_hash"] in used_map_hashes or record["map_hash"] in accepted_hashes:
            stats.duplicate += 1
            return False
        if target is not None and record["difficulty"] != target:
            stats.wrong_difficulty += 1
            return False
        accepted_hashes.add(str(record["map_hash"]))
        records.append(record)
        stats.accepted += 1
        return True

    # Stratified generation: try to meet each difficulty quota under a hard cap.
    for target in DIFFICULTIES:
        quota = quotas[target]
        per_target[target]["requested"] = quota
        max_attempts = max(1, quota * config.max_generation_attempts_per_target)
        attempts = 0
        while per_target[target]["produced"] < quota and attempts < max_attempts:
            attempts += 1
            stats.attempts += 1
            per_target[target]["attempts"] += 1
            candidate = _sample_candidate(config, rng)
            if try_accept(candidate, target):
                per_target[target]["produced"] += 1
        if per_target[target]["produced"] < quota:
            logger.warning(
                "[%s] Difficulty '%s' quota not met: %d/%d after %d attempts; "
                "remaining samples will be filled without a difficulty target.",
                split_name,
                target,
                per_target[target]["produced"],
                quota,
                attempts,
            )

    # Fill any remaining slots without a difficulty restriction, still bounded.
    remaining = count - len(records)
    fill_attempts = 0
    max_fill_attempts = max(1, remaining * config.max_generation_attempts_per_sample)
    while len(records) < count and fill_attempts < max_fill_attempts:
        fill_attempts += 1
        stats.attempts += 1
        candidate = _sample_candidate(config, rng)
        try_accept(candidate, None)
    if len(records) < count:
        raise RuntimeError(
            f"Could not generate {count} unique mazes for split '{split_name}' "
            f"after {stats.attempts} attempts "
            f"(got {len(records)}). Increase max_generation_attempts_per_sample."
        )
    if fill_attempts:
        logger.info("[%s] Filled %d remaining instances in %d attempts.", split_name, remaining, fill_attempts)

    # Shuffle while retaining all arrays in lockstep.
    order = rng.permutation(len(records))
    records = [records[int(i)] for i in order]

    difficulty_counts = {name: 0 for name in DIFFICULTIES}
    for record in records:
        difficulty_counts[_record_category(record)] += 1
    obstacle_ratios = [float(record["obstacle_ratio"]) for record in records]
    metadata = {
        "split": split_name,
        "generator_seed": int(seed),
        "requested_count": int(count),
        "actual_count": len(records),
        "generation_stats": {
            "attempts": stats.attempts,
            "accepted": stats.accepted,
            "unreachable_or_too_short": stats.unreachable,
            "duplicate_layouts": stats.duplicate,
            "wrong_difficulty_during_stratification": stats.wrong_difficulty,
        },
        "difficulty_quotas": quotas,
        "difficulty_counts": difficulty_counts,
        "per_target": per_target,
        "obstacle_ratio": {
            "min": min(obstacle_ratios) if obstacle_ratios else None,
            "max": max(obstacle_ratios) if obstacle_ratios else None,
            "mean": float(np.mean(obstacle_ratios)) if obstacle_ratios else None,
        },
        "bfs_length": {
            "min": int(min(r["bfs_length"] for r in records)) if records else None,
            "max": int(max(r["bfs_length"] for r in records)) if records else None,
            "mean": float(np.mean([r["bfs_length"] for r in records])) if records else None,
        },
    }
    return records, metadata


def _records_to_arrays(records: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if not records:
        raise ValueError("Cannot serialize an empty dataset")
    return {
        "mazes": np.stack([r["maze"] for r in records]).astype(np.int8),
        "starts": np.asarray([r["start"] for r in records], dtype=np.int16),
        "goals": np.asarray([r["goal"] for r in records], dtype=np.int16),
        "seeds": np.asarray([r["seed"] for r in records], dtype=np.int64),
        "bfs_lengths": np.asarray([r["bfs_length"] for r in records], dtype=np.int16),
        "difficulties": np.asarray([r["difficulty"] for r in records], dtype="<U8"),
        "map_hashes": np.asarray([r["map_hash"] for r in records], dtype="<U64"),
        "obstacle_ratios": np.asarray([r["obstacle_ratio"] for r in records], dtype=np.float32),
    }


def save_split(path: str | Path, records: Sequence[Dict[str, Any]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez_compressed(path, **_records_to_arrays(records))
    return path


def generate_datasets(
    config: Config,
    *,
    force: bool = False,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generate and save train/validation/test NPZ files plus manifest.

    A shared set of accepted map hashes guarantees that no obstacle layout is
    reused across splits.  The function is deterministic given the three
    dataset seeds in :class:`Config`.
    """
    ensure_dir(config.data_dir)
    ensure_dir(config.log_dir)
    if logger is None:
        logger = setup_logger("gsp.generator", Path(config.log_dir) / "generator.log")

    split_specs = (
        ("train", config.num_train_instances, config.train_dataset_seed, config.train_path),
        ("validation", config.num_validation_instances, config.validation_dataset_seed, config.validation_path),
        ("test", config.num_test_instances, config.test_dataset_seed, config.test_path),
    )
    existing = [path for _, _, _, path in split_specs if Path(path).exists()]
    if existing and not force:
        logger.info(
            "Dataset files already exist (%s). Use --force to regenerate.",
            ", ".join(str(p) for p in existing),
        )
        manifest = load_manifest(config.manifest_path) if config.manifest_path.exists() else {}
        return manifest

    logger.info("Generating datasets with config: %s", config.data_dir)
    used_map_hashes: Set[str] = set()
    split_manifests: Dict[str, Any] = {}
    all_records: Dict[str, List[Dict[str, Any]]] = {}
    for split_name, count, seed, path in split_specs:
        logger.info(
            "Generating %s split: %d mazes, master seed=%d",
            split_name,
            count,
            seed,
        )
        records, metadata = _generate_split(
            config,
            split_name,
            count,
            seed,
            used_map_hashes,
            logger,
        )
        # Track every accepted layout globally; also assert the local invariant.
        for record in records:
            map_hash = str(record["map_hash"])
            if map_hash in used_map_hashes:
                raise RuntimeError(f"Internal duplicate map hash after generation: {map_hash}")
        used_map_hashes.update(str(record["map_hash"]) for record in records)
        save_split(path, records)
        metadata["path"] = str(path)
        metadata["per_maze_seed_stored"] = True
        split_manifests[split_name] = metadata
        all_records[split_name] = records
        logger.info(
            "  saved %s (%d; difficulty=%s; BFS mean=%.2f)",
            path,
            len(records),
            metadata["difficulty_counts"],
            metadata["bfs_length"]["mean"],
        )

    independence = audit_splits(
        {
            "train": all_records["train"],
            "validation": all_records["validation"],
            "test": all_records["test"],
        }
    )
    if independence["total_duplicate_instances"] != 0:
        raise RuntimeError(f"Dataset independence check failed: {independence}")
    logger.info("Map-hash independence check passed: no layout is shared across splits.")

    manifest = {
        "format_version": 1,
        "created_at_utc": utc_now_iso(),
        "generator": "procedural binary mazes (no external data)",
        "config": config.to_dict(),
        "splits": split_manifests,
        "independence": independence,
        "state_encoding": "[obstacle_map_flat, agent_one_hot, goal_one_hot] (row-major)",
        "note": (
            "BFS lengths are metadata for difficulty and evaluation only. "
            "They are not used as Q-learning targets."
        ),
    }
    save_json(manifest, config.manifest_path)
    logger.info("Saved dataset manifest: %s", config.manifest_path)
    return manifest


def audit_splits(splits: Dict[str, Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    """Hash-based duplicate audit across named splits."""
    hash_sets: Dict[str, Set[str]] = {}
    for name, records in splits.items():
        hash_sets[name] = {str(record["map_hash"]) for record in records}
    names = list(hash_sets)
    pairwise: Dict[str, int] = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            pairwise[f"{left}__{right}"] = len(hash_sets[left] & hash_sets[right])
    total_duplicates = 0
    for name, hashes in hash_sets.items():
        total_duplicates += len(splits[name]) - len(hashes)
    total_duplicates += sum(pairwise.values())
    return {
        "checked": True,
        "pairwise_overlaps": pairwise,
        "duplicate_hashes_within_splits": {
            name: len(splits[name]) - len(hash_sets[name]) for name in names
        },
        "total_duplicate_instances": total_duplicates,
        "unique_layouts": {name: len(hash_set) for name, hash_set in hash_sets.items()},
    }


def load_manifest(path: str | Path) -> Dict[str, Any]:
    from utils import load_json

    return load_json(path)


def load_dataset(path: str | Path) -> Tuple[List[MazeInstance], Dict[str, Any]]:
    """Load a split as ``(instances, metadata_arrays)``.

    ``allow_pickle`` is deliberately disabled because the NPZ files contain
    only primitive arrays.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run 'python generator.py' first."
        )
    with np.load(path, allow_pickle=False) as data:
        mazes = np.asarray(data["mazes"], dtype=np.int8)
        starts = np.asarray(data["starts"], dtype=np.int64)
        goals = np.asarray(data["goals"], dtype=np.int64)
        seeds = np.asarray(data["seeds"], dtype=np.int64)
        bfs_lengths = np.asarray(data["bfs_lengths"], dtype=np.int64)
        difficulties = np.asarray(data["difficulties"]).astype(str)
        map_hashes = np.asarray(data["map_hashes"]).astype(str)
        obstacle_ratios = np.asarray(data["obstacle_ratios"], dtype=np.float32)

    n = len(mazes)
    required_lengths = {
        "starts": len(starts),
        "goals": len(goals),
        "seeds": len(seeds),
        "bfs_lengths": len(bfs_lengths),
        "difficulties": len(difficulties),
        "map_hashes": len(map_hashes),
        "obstacle_ratios": len(obstacle_ratios),
    }
    if any(length != n for length in required_lengths.values()):
        raise ValueError(f"Inconsistent arrays in dataset {path}: {required_lengths}, mazes={n}")

    instances: List[MazeInstance] = []
    for index in range(n):
        maze = validate_maze(mazes[index])
        start = (int(starts[index, 0]), int(starts[index, 1]))
        goal = (int(goals[index, 0]), int(goals[index, 1]))
        if not is_free(maze, start) or not is_free(maze, goal):
            raise ValueError(f"Dataset {path} contains an obstacle start/goal at index {index}")
        instances.append(
            MazeInstance(
                maze=maze.copy(),
                start=start,
                goal=goal,
                instance_id=index,
                bfs_length=int(bfs_lengths[index]),
                difficulty=str(difficulties[index]),
                seed=int(seeds[index]),
                map_hash=str(map_hashes[index]),
            )
        )
    metadata = {
        "path": str(path),
        "count": n,
        "mazes": mazes,
        "starts": starts,
        "goals": goals,
        "seeds": seeds,
        "bfs_lengths": bfs_lengths,
        "difficulties": difficulties,
        "map_hashes": map_hashes,
        "obstacle_ratios": obstacle_ratios,
    }
    return instances, metadata


def load_dataset_instances(path: str | Path) -> List[MazeInstance]:
    return load_dataset(path)[0]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate procedural maze datasets for GSP")
    parser.add_argument("--force", action="store_true", help="overwrite existing dataset files")
    parser.add_argument("--data-dir", type=str, default=None, help="override config data directory")
    args = parser.parse_args(argv)

    config = Config()
    if args.data_dir is not None:
        config = config.copy_with(data_dir=args.data_dir)
    logger = setup_logger("gsp.generator", Path(config.log_dir) / "generator.log")
    manifest = generate_datasets(config, force=args.force, logger=logger)
    logger.info("Generation complete. Splits: %s", list(manifest.get("splits", {})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())