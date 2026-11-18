"""Dataset generation, metadata and split-independence tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from config import Config
from environment import bfs_shortest_path, is_free
from generator import DIFFICULTIES, generate_datasets, load_dataset


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def test_required_datasets_exist_with_expected_counts() -> None:
    config = Config()
    if not all(path.exists() for path in (config.train_path, config.validation_path, config.test_path)):
        pytest.skip("Run 'python generator.py --force' before the full dataset tests")
    train, _ = load_dataset(config.train_path)
    validation, _ = load_dataset(config.validation_path)
    test, _ = load_dataset(config.test_path)
    assert len(train) == 1000
    assert len(validation) == 200
    assert len(test) == 300


def test_dataset_starts_goals_hashes_and_difficulty_metadata() -> None:
    config = Config()
    if not config.manifest_path.exists():
        pytest.skip("Generate the datasets first")
    all_instances = []
    all_hashes: list[str] = []
    for path in (config.train_path, config.validation_path, config.test_path):
        instances, metadata = load_dataset(path)
        all_instances.extend(instances)
        all_hashes.extend(str(value) for value in metadata["map_hashes"])
        for instance in instances:
            assert instance.maze.shape == (8, 8)
            assert set(np.unique(instance.maze)).issubset({0, 1})
            assert is_free(instance.maze, instance.start)
            assert is_free(instance.maze, instance.goal)
            assert instance.start != instance.goal
            assert instance.seed is not None
            assert instance.map_hash
            assert instance.difficulty in DIFFICULTIES
            assert instance.bfs_length is not None
            if instance.bfs_length <= 6:
                assert instance.difficulty == "easy"
            elif instance.bfs_length <= 10:
                assert instance.difficulty == "medium"
            else:
                assert instance.difficulty == "hard"
            ratio = float(np.mean(instance.maze))
            assert 0.10 - 1e-6 <= ratio <= 0.25 + 1e-6
    assert len(all_instances) == 1500
    assert len(all_hashes) == len(set(all_hashes)), "Map layouts must not repeat across splits"


def test_stored_bfs_lengths_are_correct() -> None:
    config = Config()
    if not config.test_path.exists():
        pytest.skip("Generate the datasets first")
    # Recompute shortest paths for every split; BFS is metadata, not a
    # Q-learning label, so checking it here cannot leak into training.
    for path in (config.train_path, config.validation_path, config.test_path):
        instances, _ = load_dataset(path)
        for instance in instances:
            shortest_path = bfs_shortest_path(
                instance.maze, instance.start, instance.goal
            )
            assert shortest_path is not None
            assert len(shortest_path) - 1 == instance.bfs_length


def test_manifest_master_seeds_are_distinct_and_independence_checked() -> None:
    config = Config()
    if not config.manifest_path.exists():
        pytest.skip("Generate the datasets first")
    from utils import load_json

    manifest = load_json(config.manifest_path)
    seeds = {
        manifest["splits"][name]["generator_seed"]
        for name in ("train", "validation", "test")
    }
    assert len(seeds) == 3
    independence = manifest["independence"]
    assert independence["checked"] is True
    assert independence["total_duplicate_instances"] == 0
    assert all(value == 0 for value in independence["pairwise_overlaps"].values())
    distribution = manifest["splits"]["train"]["difficulty_counts"]
    assert distribution["easy"] > 0
    assert distribution["medium"] > 0
    assert distribution["hard"] > 0


def test_small_generation_is_reproducible(tmp_path: Path) -> None:
    config = Config().copy_with(
        num_train_instances=9,
        num_validation_instances=6,
        num_test_instances=6,
        data_dir=str(tmp_path / "data"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
    )
    generate_datasets(config, force=True)
    first = {}
    for name, path in (
        ("train", config.train_path),
        ("validation", config.validation_path),
        ("test", config.test_path),
    ):
        instances, metadata = load_dataset(path)
        first[name] = metadata["map_hashes"].copy()
        assert len(instances) == {
            "train": 9,
            "validation": 6,
            "test": 6,
        }[name]

    generate_datasets(config, force=True)
    for name, path in (
        ("train", config.train_path),
        ("validation", config.validation_path),
        ("test", config.test_path),
    ):
        _, metadata = load_dataset(path)
        np.testing.assert_array_equal(first[name], metadata["map_hashes"])
    assert config.manifest_path.exists()
    assert config.train_path.exists() and config.validation_path.exists() and config.test_path.exists()