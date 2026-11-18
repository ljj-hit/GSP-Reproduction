"""Main GSP training loop.

One episode = select one training maze + run one WA* search + add its
transitions to the shared replay buffer + run up to two gradient updates.
Weights are never modified while a search is in progress.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from check_cuda import assert_batch_on_cuda, assert_model_on_cuda, require_cuda
from config import Config
from generator import load_dataset
from model import build_q_network
from replay_buffer import ReplayBuffer
from search import QGuidedWAStar, SearchResult
from trainer import (
    InstancePoolSampler,
    QTrainer,
    UpdateStats,
    ValidationMetrics,
    validate_network,
)
from utils import (
    ensure_dir,
    format_seconds,
    save_checkpoint,
    save_json,
    set_seed,
    setup_logger,
    utc_now_iso,
)


TRAINING_LOG_COLUMNS = (
    "episode",
    "instance_id",
    "instance_category",
    "success",
    "expansions",
    "plan_length",
    "buffer_size",
    "loss",
    "q_mean",
    "q_min",
    "q_max",
    "episode_time",
    "gradient_norm",
    "gradient_updates",
    "target_updates",
    "failure_reason",
)

VALIDATION_LOG_COLUMNS = (
    "episode",
    "success_rate",
    "mean_expansions_all",
    "mean_expansions_solved",
    "mean_plan_length_solved",
    "num_solved",
    "num_instances",
    "is_best",
    "elapsed_seconds",
)

UPDATE_LOG_COLUMNS = (
    "episode",
    "update_index",
    "loss",
    "q_mean",
    "q_min",
    "q_max",
    "gradient_norm",
    "target_network_updated",
    "gradient_updates_total",
)


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(values))


def _check_dataset_sizes(config: Config, num_train: int, num_validation: int) -> None:
    if num_train != config.num_train_instances:
        raise ValueError(
            f"Train dataset has {num_train} mazes but config expects "
            f"{config.num_train_instances}. Regenerate data or update config."
        )
    if num_validation != config.num_validation_instances:
        raise ValueError(
            f"Validation dataset has {num_validation} mazes but config expects "
            f"{config.num_validation_instances}. Regenerate data or update config."
        )


def _check_train_validation_disjoint(train_hashes: Sequence[str], validation_hashes: Sequence[str]) -> None:
    overlap = set(map(str, train_hashes)) & set(map(str, validation_hashes))
    if overlap:
        raise RuntimeError(
            f"Train/validation map-hash overlap detected ({len(overlap)} layouts). "
            "Regenerate the datasets."
        )
    if len(set(map(str, train_hashes))) != len(train_hashes):
        raise RuntimeError("Duplicate layouts exist inside the training split")
    if len(set(map(str, validation_hashes))) != len(validation_hashes):
        raise RuntimeError("Duplicate layouts exist inside the validation split")


def _aggregate_updates(stats: Sequence[UpdateStats]) -> Dict[str, Optional[float]]:
    if not stats:
        return {
            "loss": None,
            "q_mean": None,
            "q_min": None,
            "q_max": None,
            "gradient_norm": None,
        }
    return {
        "loss": _mean_or_none([entry.loss for entry in stats]),
        "q_mean": _mean_or_none([entry.q_mean for entry in stats]),
        "q_min": min(entry.q_min for entry in stats),
        "q_max": max(entry.q_max for entry in stats),
        "gradient_norm": _mean_or_none([entry.gradient_norm for entry in stats]),
    }


def _write_training_row(writer: Any, row: Dict[str, Any]) -> None:
    writer.writerow({column: row.get(column, "") for column in TRAINING_LOG_COLUMNS})


def _write_validation_row(writer: Any, row: Dict[str, Any]) -> None:
    writer.writerow({column: row.get(column, "") for column in VALIDATION_LOG_COLUMNS})


def _save_model_checkpoint(
    config: Config,
    path: Path,
    trainer: QTrainer,
    episode: int,
    validation: Optional[ValidationMetrics],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload_extra: Dict[str, Any] = {
        "seed": config.training_seed,
        "gradient_updates": trainer.gradient_updates,
        "target_updates": trainer.target_updates,
    }
    if validation is not None:
        payload_extra["validation_metrics"] = validation.as_dict()
    if extra:
        payload_extra.update(extra)
    save_checkpoint(
        path,
        trainer.model,
        target_model=trainer.target_model,
        optimizer=trainer.optimizer,
        episode=episode,
        config=config,
        extra=payload_extra,
    )


def run_training(config: Config) -> Dict[str, Any]:
    """Run a complete single-seed training job.

    The test split is deliberately never opened here.
    """
    config.validate()
    set_seed(config.training_seed, deterministic=True)
    device = require_cuda(config.device, verbose=True)
    ensure_dir(config.data_dir)
    ensure_dir(config.checkpoint_dir)
    ensure_dir(config.log_dir)
    ensure_dir(config.results_dir)
    ensure_dir(config.figures_dir)

    logger = setup_logger(
        "gsp.train",
        Path(config.log_dir) / "training.log",
    )
    config_path = config.save_json(Path(config.checkpoint_dir) / "config.json")

    train_instances, train_metadata = load_dataset(config.train_path)
    validation_instances, validation_metadata = load_dataset(config.validation_path)
    _check_dataset_sizes(config, len(train_instances), len(validation_instances))
    _check_train_validation_disjoint(
        train_metadata["map_hashes"].tolist(),
        validation_metadata["map_hashes"].tolist(),
    )

    model = build_q_network(config).to(device)
    assert_model_on_cuda(model, device)
    trainer = QTrainer(config, device=device, model=model)
    assert_model_on_cuda(trainer.model, device)
    assert_model_on_cuda(trainer.target_model, device)
    startup_probe_batch = {
        "states": torch.zeros((1, config.state_dim), dtype=torch.float32, device=device),
        "actions": torch.zeros((1,), dtype=torch.long, device=device),
        "next_states": torch.zeros((1, config.state_dim), dtype=torch.float32, device=device),
        "return_bounds": torch.full((1,), -float("inf"), dtype=torch.float32, device=device),
        "terminal": torch.zeros((1,), dtype=torch.bool, device=device),
        "dead_end": torch.zeros((1,), dtype=torch.bool, device=device),
        "next_valid_mask": torch.ones((1, 4), dtype=torch.bool, device=device),
    }
    assert_batch_on_cuda(startup_probe_batch, device)

    replay = ReplayBuffer(config.replay_capacity, seed=config.training_seed + 10_001)
    sampler = InstancePoolSampler(
        config,
        [instance.instance_id for instance in train_instances],
        seed=config.training_seed + 20_003,
    )
    search = QGuidedWAStar(model, config, device)

    logger.info("=" * 78)
    logger.info("GSP training start")
    logger.info("Seed: %d", config.training_seed)
    logger.info("CUDA device: %s (%s)", device, torch.cuda.get_device_name(device))
    logger.info("Online model parameter device: %s", next(model.parameters()).device)
    logger.info("Target model parameter device: %s", next(trainer.target_model.parameters()).device)
    logger.info("Training Batch tensor device: %s", startup_probe_batch["states"].device)
    logger.info("Training mazes: %d | validation mazes: %d", len(train_instances), len(validation_instances))
    logger.info("Episodes: %d | search budget: %d | WA* weight: %.2f", config.episodes, config.search_budget, config.wa_star_weight)
    logger.info("Replay capacity: %d | batch: %d | updates/episode: %d", config.replay_capacity, config.batch_size, config.updates_per_episode)
    logger.info("Solution bounds enabled: %s | dynamic sampling: %s", config.use_solution_bounds, config.use_instance_sampling)
    logger.info("Config saved to: %s", config_path)
    logger.info("=" * 78)

    training_log_path = config.training_log_path
    validation_log_path = config.validation_log_path
    update_log_path = Path(config.log_dir) / "update_log.csv"
    ensure_dir(training_log_path.parent)
    ensure_dir(validation_log_path.parent)
    ensure_dir(update_log_path.parent)

    best_metric: Optional[tuple] = None
    best_validation: Optional[ValidationMetrics] = None
    best_episode: Optional[int] = None
    validation_history: List[Dict[str, Any]] = []
    last_validation: Optional[ValidationMetrics] = None
    total_gradient_updates = 0
    target_sync_count = 0
    recent: Deque[Tuple[bool, int]] = deque(maxlen=100)
    training_start = time.perf_counter()
    batch_device_printed = False
    final_loss: Optional[float] = None

    with training_log_path.open("w", newline="", encoding="utf-8") as train_handle, \
            validation_log_path.open("w", newline="", encoding="utf-8") as validation_handle, \
            update_log_path.open("w", newline="", encoding="utf-8") as update_handle:
        train_writer = csv.DictWriter(train_handle, fieldnames=TRAINING_LOG_COLUMNS)
        validation_writer = csv.DictWriter(validation_handle, fieldnames=VALIDATION_LOG_COLUMNS)
        update_writer = csv.DictWriter(update_handle, fieldnames=UPDATE_LOG_COLUMNS)
        train_writer.writeheader()
        validation_writer.writeheader()
        update_writer.writeheader()

        progress = tqdm(range(1, config.episodes + 1), desc=f"GSP seed {config.training_seed}", unit="ep")
        for episode in progress:
            episode_start = time.perf_counter()
            instance_id = sampler.choose()
            instance = train_instances[instance_id]

            # Frozen/no-grad inference inside search; no optimizer step until done.
            result: SearchResult = search.search_instance(instance)
            category = InstancePoolSampler.classify(result)
            sampler.update(instance.instance_id, category)
            replay.add_many(result.experiences)

            update_stats: List[UpdateStats] = []
            if len(replay) >= config.batch_size:
                for _ in range(config.updates_per_episode):
                    batch = replay.sample(config.batch_size, device=device)
                    if not batch_device_printed:
                        assert_batch_on_cuda(batch, device)
                        report = trainer.full_batch_device_report(batch)
                        logger.info("First training batch device audit: %s", report)
                        logger.info(
                            "Online parameter device check: %s",
                            "cuda" if next(trainer.model.parameters()).device.type == "cuda" else "NOT CUDA",
                        )
                        batch_device_printed = True
                    stats = trainer.train_step(batch)
                    update_stats.append(stats)
                    update_writer.writerow(
                        {
                            "episode": episode,
                            "update_index": len(update_stats),
                            "loss": f"{stats.loss:.10f}",
                            "q_mean": f"{stats.q_mean:.10f}",
                            "q_min": f"{stats.q_min:.10f}",
                            "q_max": f"{stats.q_max:.10f}",
                            "gradient_norm": f"{stats.gradient_norm:.10f}",
                            "target_network_updated": int(stats.target_network_updated),
                            "gradient_updates_total": stats.gradient_updates,
                        }
                    )
                    update_handle.flush()
                    if stats.target_network_updated:
                        target_sync_count += 1

            total_gradient_updates = trainer.gradient_updates
            aggregate = _aggregate_updates(update_stats)
            final_loss = aggregate["loss"] if aggregate["loss"] is not None else final_loss
            episode_time = time.perf_counter() - episode_start
            recent.append((result.success, result.expansions))

            train_row = {
                "episode": episode,
                "instance_id": instance.instance_id,
                "instance_category": category,
                "success": int(result.success),
                "expansions": result.expansions,
                "plan_length": result.plan_length if result.success else "",
                "buffer_size": len(replay),
                "loss": "" if aggregate["loss"] is None else f"{aggregate['loss']:.8f}",
                "q_mean": "" if aggregate["q_mean"] is None else f"{aggregate['q_mean']:.8f}",
                "q_min": "" if aggregate["q_min"] is None else f"{aggregate['q_min']:.8f}",
                "q_max": "" if aggregate["q_max"] is None else f"{aggregate['q_max']:.8f}",
                "episode_time": f"{episode_time:.6f}",
                "gradient_norm": "" if aggregate["gradient_norm"] is None else f"{aggregate['gradient_norm']:.8f}",
                "gradient_updates": total_gradient_updates,
                "target_updates": target_sync_count,
                "failure_reason": "" if result.success else result.termination_reason,
            }
            _write_training_row(train_writer, train_row)
            train_handle.flush()

            recent_success = sum(1 for solved, _ in recent if solved) / len(recent)
            recent_expansions = sum(exp for _, exp in recent) / len(recent)
            loss_text = "n/a" if aggregate["loss"] is None else f"{aggregate['loss']:.4f}"
            progress.set_postfix(
                {
                    "succ": f"{recent_success:.2f}",
                    "exp": f"{recent_expansions:.1f}",
                    "loss": loss_text,
                    "buf": len(replay),
                    "upd": total_gradient_updates,
                },
                refresh=False,
            )
            if episode == 1 or episode % 25 == 0:
                logger.info(
                    "Episode %d/%d | recent success=%.3f | recent avg expansions=%.2f | "
                    "loss=%s | buffer=%d | updates=%d | category=%s | elapsed=%s | CUDA=%s",
                    episode,
                    config.episodes,
                    recent_success,
                    recent_expansions,
                    loss_text,
                    len(replay),
                    total_gradient_updates,
                    category,
                    format_seconds(time.perf_counter() - training_start),
                    torch.cuda.get_device_name(device),
                )

            should_validate = (
                episode % config.validation_interval == 0 or episode == config.episodes
            )
            if should_validate:
                validation_start = time.perf_counter()
                validation = validate_network(model, validation_instances, config, device)
                validation_elapsed = time.perf_counter() - validation_start
                last_validation = validation
                key = validation.selection_key()
                is_best = best_metric is None or key > best_metric
                if is_best:
                    best_metric = key
                    best_validation = validation
                    best_episode = episode
                    _save_model_checkpoint(
                        config,
                        config.best_checkpoint_path,
                        trainer,
                        episode,
                        validation,
                        extra={"selection_key": list(key), "is_best": True},
                    )
                _save_model_checkpoint(
                    config,
                    config.last_checkpoint_path,
                    trainer,
                    episode,
                    validation,
                    extra={"is_best": is_best},
                )
                validation_row = {
                    "episode": episode,
                    "success_rate": f"{validation.success_rate:.8f}",
                    "mean_expansions_all": f"{validation.mean_expansions_all:.8f}",
                    "mean_expansions_solved": (
                        "" if not np.isfinite(validation.mean_expansions_solved)
                        else f"{validation.mean_expansions_solved:.8f}"
                    ),
                    "mean_plan_length_solved": (
                        "" if not np.isfinite(validation.mean_plan_length_solved)
                        else f"{validation.mean_plan_length_solved:.8f}"
                    ),
                    "num_solved": validation.num_solved,
                    "num_instances": validation.num_instances,
                    "is_best": int(is_best),
                    "elapsed_seconds": f"{time.perf_counter() - training_start:.6f}",
                }
                _write_validation_row(validation_writer, validation_row)
                validation_handle.flush()
                validation_history.append(
                    {
                        "episode": episode,
                        **validation.as_dict(),
                        "is_best": is_best,
                        "validation_time_seconds": validation_elapsed,
                    }
                )
                logger.info(
                    "Validation @ episode %d | success=%.4f (%d/%d) | "
                    "avg expansions(all)=%.2f | avg expansions(solved)=%s | "
                    "best=%s | val time=%s",
                    episode,
                    validation.success_rate,
                    validation.num_solved,
                    validation.num_instances,
                    validation.mean_expansions_all,
                    "n/a"
                    if not np.isfinite(validation.mean_expansions_solved)
                    else f"{validation.mean_expansions_solved:.2f}",
                    is_best,
                    format_seconds(validation_elapsed),
                )

        # The final episode is always validated above.  This fallback only
        # triggers for an unusual interruption of that invariant.
        if last_validation is None:
            validation = validate_network(model, validation_instances, config, device)
            last_validation = validation
            key = validation.selection_key()
            is_best = best_metric is None or key > best_metric
            if is_best:
                best_metric = key
                best_validation = validation
                best_episode = config.episodes
                _save_model_checkpoint(
                    config,
                    config.best_checkpoint_path,
                    trainer,
                    config.episodes,
                    validation,
                    extra={"selection_key": list(key), "is_best": True},
                )
            _save_model_checkpoint(
                config,
                config.last_checkpoint_path,
                trainer,
                config.episodes,
                validation,
                extra={"is_best": is_best},
            )

    progress.close()
    elapsed = time.perf_counter() - training_start
    if best_validation is None:
        best_validation = validate_network(model, validation_instances, config, device)
        best_episode = config.episodes
    summary = {
        "status": "completed",
        "created_at_utc": utc_now_iso(),
        "seed": config.training_seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "episodes": config.episodes,
        "elapsed_seconds": elapsed,
        "gradient_updates": total_gradient_updates,
        "target_updates": target_sync_count,
        "replay_size": len(replay),
        "last_loss": final_loss,
        "best_episode": best_episode,
        "best_validation": None if best_validation is None else best_validation.as_dict(),
        "validation_history": validation_history,
        "config_path": str(config_path),
        "best_checkpoint": str(config.best_checkpoint_path),
        "last_checkpoint": str(config.last_checkpoint_path),
        "training_log": str(training_log_path),
        "validation_log": str(validation_log_path),
        "update_log": str(update_log_path),
        "training_log_columns": list(TRAINING_LOG_COLUMNS),
        "validation_log_columns": list(VALIDATION_LOG_COLUMNS),
        "update_log_columns": list(UPDATE_LOG_COLUMNS),
    }
    logger.info("Training complete in %s", format_seconds(elapsed))
    logger.info("Best validation: %s", None if best_validation is None else best_validation.as_dict())
    logger.info("Best checkpoint: %s", config.best_checkpoint_path)
    logger.info("Last checkpoint: %s", config.last_checkpoint_path)
    save_json(summary, Path(config.checkpoint_dir) / "training_summary.json")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GSP Q-network-guided WA* on CUDA")
    parser.add_argument("--device", type=str, default=None, help="CUDA device, e.g. cuda:0")
    parser.add_argument("--seed", type=int, default=None, help="training random seed")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--search-budget", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--updates-per-episode", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--target-update-interval", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=None)
    parser.add_argument(
        "--no-solution-bounds",
        action="store_true",
        help="ablation: disable search-derived finite return bounds",
    )
    parser.add_argument(
        "--uniform-sampling",
        action="store_true",
        help="ablation: disable dynamic instance-pool sampling",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    overrides: Dict[str, Any] = {}
    for attribute in (
        "device",
        "episodes",
        "data_dir",
        "checkpoint_dir",
        "log_dir",
        "results_dir",
        "search_budget",
        "batch_size",
        "updates_per_episode",
        "validation_interval",
        "target_update_interval",
        "learning_rate",
    ):
        value = getattr(args, attribute)
        if value is not None:
            overrides[attribute] = value
    if args.seed is not None:
        if args.seed < 0:
            raise ValueError("Seed must be non-negative")
        overrides["training_seed"] = int(args.seed)
    if args.hidden_dims is not None:
        overrides["hidden_dims"] = tuple(int(value) for value in args.hidden_dims)
    if args.no_solution_bounds:
        overrides["use_solution_bounds"] = False
    if args.uniform_sampling:
        overrides["use_instance_sampling"] = False
    return Config().copy_with(**overrides)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    config.validate()
    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())