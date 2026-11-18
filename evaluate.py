"""Evaluate a trained GSP model and all baselines on the held-out test set.

The test split is opened only here (and by the experiment runner).  No model
parameter update, replay-buffer write, dataset modification, or test-guided
tuning occurs during evaluation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from baselines import (
    GreedyQPolicy,
    run_astar,
    run_bfs,
    run_weighted_astar,
)
from check_cuda import assert_model_on_cuda, require_cuda
from config import Config
from environment import MazeInstance, encode_state
from generator import load_dataset
from metrics import (
    save_summary,
    summarize_records,
    write_evaluation_csv,
)
from model import build_q_network
from search import QGuidedWAStar, SearchResult
from utils import (
    ensure_dir,
    format_seconds,
    load_checkpoint,
    set_seed,
    setup_logger,
    utc_now_iso,
)


BFS_NAME = "BFS"
ASTAR_NAME = "A* + Manhattan"
WASTAR_NAME = "WA* + Manhattan"
UNTRAINED_GSP_NAME = "Untrained-Q WA*"
GSP_NAME = "GSP-WA*"
GREEDY_NAME = "Greedy-Q"
MAIN_ALGORITHMS = (BFS_NAME, ASTAR_NAME, WASTAR_NAME, UNTRAINED_GSP_NAME, GSP_NAME)


def _run_timed(
    function: Callable[[], SearchResult],
    *,
    synchronize_device: Optional[torch.device] = None,
) -> Tuple[SearchResult, float]:
    if synchronize_device is not None and synchronize_device.type == "cuda":
        torch.cuda.synchronize(synchronize_device)
    start = time.perf_counter()
    result = function()
    if synchronize_device is not None and synchronize_device.type == "cuda":
        torch.cuda.synchronize(synchronize_device)
    return result, time.perf_counter() - start


def _warmup_model(model: torch.nn.Module, instance: MazeInstance, device: torch.device) -> None:
    state = encode_state(instance.maze, instance.start, instance.goal)
    tensor = torch.from_numpy(state).to(device, dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_record(
    algorithm: str,
    instance: MazeInstance,
    result: SearchResult,
    total_time: float,
    bfs_length: int,
    seed: int,
) -> Dict[str, Any]:
    uses_network = algorithm in {UNTRAINED_GSP_NAME, GSP_NAME, GREEDY_NAME}
    inference_time = result.inference_time_seconds if uses_network else 0.0
    non_inference_time = max(0.0, total_time - inference_time)
    path_ratio: Optional[float]
    if result.success and bfs_length > 0:
        path_ratio = result.plan_length / bfs_length
    elif result.success and bfs_length == 0:
        path_ratio = 1.0
    else:
        path_ratio = None
    return {
        "algorithm": algorithm,
        "instance_id": instance.instance_id,
        "difficulty": instance.difficulty,
        "bfs_length": bfs_length,
        "success": bool(result.success),
        "expansions": int(result.expansions),
        "plan_length": int(result.plan_length) if result.success else None,
        "path_ratio": path_ratio,
        "search_time_seconds": float(total_time),
        "inference_time_seconds": float(inference_time),
        "non_inference_time_seconds": float(non_inference_time),
        "failure_reason": "" if result.success else result.termination_reason,
        "seed": seed,
    }


def evaluate_checkpoint(
    config: Config,
    checkpoint_path: str | Path,
    instances: Sequence[MazeInstance],
    *,
    device: str | torch.device,
    output_dir: str | Path,
    include_greedy: Optional[bool] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full held-out comparison and write CSV/JSON artifacts."""
    if len(instances) == 0:
        raise ValueError("Evaluation requires at least one test instance")
    selected_device = require_cuda(device, verbose=verbose)
    # Runtime device takes precedence, but the stored config must still
    # describe a CUDA experiment.
    if str(config.device) != str(selected_device):
        config = config.copy_with(device=str(selected_device))
    output_dir = ensure_dir(output_dir)
    logger = setup_logger("gsp.evaluate", Path(output_dir) / "evaluation.log")
    set_seed(config.training_seed, deterministic=True)

    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    trained_model = build_q_network(config)
    missing_unexpected = trained_model.load_state_dict(payload["model_state_dict"], strict=True)
    del missing_unexpected
    trained_model.to(selected_device)
    trained_model.eval()
    assert_model_on_cuda(trained_model, selected_device)

    # A separately initialised network is the random-Q control.  Seeding with
    # the run seed makes this control deterministic and traceable.
    set_seed(config.training_seed + 700_001, deterministic=True)
    untrained_model = build_q_network(config).to(selected_device)
    untrained_model.eval()
    assert_model_on_cuda(untrained_model, selected_device)

    first_instance = instances[0]
    _warmup_model(trained_model, first_instance, selected_device)
    _warmup_model(untrained_model, first_instance, selected_device)

    trained_search = QGuidedWAStar(trained_model, config, selected_device)
    untrained_search = QGuidedWAStar(untrained_model, config, selected_device)
    greedy_enabled = config.include_greedy_evaluation if include_greedy is None else include_greedy
    greedy_policy = (
        GreedyQPolicy(
            trained_model,
            config,
            selected_device,
            max_steps=config.greedy_max_steps,
        )
        if greedy_enabled
        else None
    )

    records: List[Dict[str, Any]] = []
    total_steps = len(instances) * (len(MAIN_ALGORITHMS) + int(greedy_enabled))
    progress = tqdm(total=total_steps, desc="Test evaluation", unit="run")
    evaluation_start = time.perf_counter()
    try:
        for instance in instances:
            bfs_result, bfs_time = _run_timed(
                lambda: run_bfs(
                    instance.maze,
                    instance.start,
                    instance.goal,
                    max_expansions=config.search_budget,
                )
            )
            if not bfs_result.success:
                raise RuntimeError(
                    f"BFS failed on a generated test instance {instance.instance_id}; "
                    "the dataset is inconsistent."
                )
            bfs_length = bfs_result.plan_length
            if instance.bfs_length is not None and bfs_length != int(instance.bfs_length):
                raise RuntimeError(
                    f"Stored BFS length ({instance.bfs_length}) disagrees with recomputed "
                    f"shortest path ({bfs_length}) for instance {instance.instance_id}"
                )
            records.append(
                _make_record(
                    BFS_NAME,
                    instance,
                    bfs_result,
                    bfs_time,
                    bfs_length,
                    config.training_seed,
                )
            )
            progress.update(1)

            astar_result, astar_time = _run_timed(
                lambda: run_astar(
                    instance.maze,
                    instance.start,
                    instance.goal,
                    max_expansions=config.search_budget,
                )
            )
            records.append(
                _make_record(
                    ASTAR_NAME,
                    instance,
                    astar_result,
                    astar_time,
                    bfs_length,
                    config.training_seed,
                )
            )
            progress.update(1)

            wastar_result, wastar_time = _run_timed(
                lambda: run_weighted_astar(
                    instance.maze,
                    instance.start,
                    instance.goal,
                    weight=config.wa_star_weight,
                    max_expansions=config.search_budget,
                )
            )
            records.append(
                _make_record(
                    WASTAR_NAME,
                    instance,
                    wastar_result,
                    wastar_time,
                    bfs_length,
                    config.training_seed,
                )
            )
            progress.update(1)

            untrained_result, untrained_time = _run_timed(
                lambda: untrained_search.search_instance(instance),
                synchronize_device=selected_device,
            )
            records.append(
                _make_record(
                    UNTRAINED_GSP_NAME,
                    instance,
                    untrained_result,
                    untrained_time,
                    bfs_length,
                    config.training_seed,
                )
            )
            progress.update(1)

            trained_result, trained_time = _run_timed(
                lambda: trained_search.search_instance(instance),
                synchronize_device=selected_device,
            )
            records.append(
                _make_record(
                    GSP_NAME,
                    instance,
                    trained_result,
                    trained_time,
                    bfs_length,
                    config.training_seed,
                )
            )
            progress.update(1)

            if greedy_policy is not None:
                greedy_result, greedy_time = _run_timed(
                    lambda: greedy_policy.rollout(
                        instance.maze, instance.start, instance.goal
                    ),
                    synchronize_device=selected_device,
                )
                records.append(
                    _make_record(
                        GREEDY_NAME,
                        instance,
                        greedy_result,
                        greedy_time,
                        bfs_length,
                        config.training_seed,
                    )
                )
                progress.update(1)
    finally:
        progress.close()

    elapsed = time.perf_counter() - evaluation_start
    algorithm_summaries = summarize_records(records)
    evaluation_csv = write_evaluation_csv(records, Path(output_dir) / "evaluation.csv")
    summary = {
        "status": "completed",
        "created_at_utc": utc_now_iso(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_episode": payload.get("episode"),
        "checkpoint_metadata": {
            key: payload[key]
            for key in ("seed", "gradient_updates", "target_updates", "validation_metrics")
            if key in payload
        },
        "test_dataset": str(config.test_path),
        "num_instances": len(instances),
        "seed": config.training_seed,
        "device": str(selected_device),
        "gpu_name": torch.cuda.get_device_name(selected_device),
        "search_budget": config.search_budget,
        "wa_star_weight": config.wa_star_weight,
        "untrained_initialization_seed": config.training_seed + 700_001,
        "evaluation_elapsed_seconds": elapsed,
        "metrics_definition": {
            "mean_expansions_all": "mean over all test instances, including failures",
            "mean_expansions_successful": "mean over solved test instances",
            "path_ratio": "plan_length / BFS shortest length, solved instances only",
            "search_time_seconds": "end-to-end wall time per instance",
            "inference_time_seconds": "Q-network forward path for learned methods",
            "non_inference_time_seconds": "total search time minus inference time",
        },
        "algorithms": algorithm_summaries,
        "evaluation_csv": str(evaluation_csv),
        "config": config.to_dict(),
    }
    summary_path = save_summary(summary, Path(output_dir) / "summary.json")
    logger.info("Evaluation complete for %d instances in %s", len(instances), format_seconds(elapsed))
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("%-22s %9s %12s %14s %12s", "Algorithm", "Success", "Exp(all)", "Exp(solved)", "Ratio")
    for algorithm, entry in algorithm_summaries.items():
        ratio = entry["mean_path_ratio_successful"]
        logger.info(
            "%-22s %9.4f %12.2f %14s %12s",
            algorithm,
            entry["success_rate"],
            entry["mean_expansions_all"],
            "n/a"
            if entry["mean_expansions_successful"] is None
            else f"{entry['mean_expansions_successful']:.2f}",
            "n/a" if ratio is None else f"{ratio:.4f}",
        )
    logger.info("Saved evaluation CSV: %s", evaluation_csv)
    logger.info("Saved summary JSON: %s", summary_path)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained GSP on the held-out test set")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--include-greedy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="also evaluate direct argmax-Q policy with cycle detection",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = load_checkpoint(args.checkpoint, map_location="cpu")
    stored_config = payload.get("config")
    config = Config.from_dict(stored_config) if isinstance(stored_config, dict) else Config()
    if args.data_dir is not None:
        config = config.copy_with(data_dir=args.data_dir)
    config = config.copy_with(device=args.device, results_dir=args.output_dir)
    instances, _ = load_dataset(config.test_path)
    if len(instances) != config.num_test_instances:
        raise ValueError(
            f"Test dataset has {len(instances)} instances; config expects "
            f"{config.num_test_instances}."
        )
    summary = evaluate_checkpoint(
        config,
        args.checkpoint,
        instances,
        device=args.device,
        output_dir=args.output_dir,
        include_greedy=args.include_greedy,
    )
    print(f"Evaluation summary: {Path(args.output_dir) / 'summary.json'}")
    print(f"Per-instance results: {Path(args.output_dir) / 'evaluation.csv'}")
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())