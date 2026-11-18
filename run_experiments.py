"""Unified experiment entry point.

Supported workflows:
  * CUDA check
  * dataset generation
  * one training run
  * the formal three-seed experiment (sequential, never parallel)
  * held-out evaluation
  * ablations isolated from the main results
  * figure generation
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from check_cuda import require_cuda
from config import Config
from evaluate import evaluate_checkpoint
from generator import generate_datasets, load_dataset
from metrics import (
    read_evaluation_csv,
    save_summary,
    seed_level_metric_aggregate,
    write_evaluation_csv,
)
from train import run_training
from utils import ensure_dir, load_checkpoint, save_json, utc_now_iso
from visualize import visualize


ABLATION_CHOICES = ("no_solution_bounds", "uniform_sampling", "mlp_capacity")


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _datasets_present(config: Config) -> bool:
    return all(
        path.is_file()
        for path in (config.train_path, config.validation_path, config.test_path, config.manifest_path)
    )


def _ensure_datasets(config: Config) -> None:
    if _datasets_present(config):
        return
    print("Dataset files are incomplete; generating all splits now.")
    generate_datasets(config, force=True)


def _single_train(args: argparse.Namespace) -> int:
    overrides: Dict[str, Any] = {}
    for field in ("device", "episodes", "data_dir", "checkpoint_dir", "log_dir", "results_dir"):
        value = getattr(args, field, None)
        if value is not None:
            overrides[field] = value
    if args.seed is not None:
        overrides["training_seed"] = args.seed
    if args.validation_interval is not None:
        overrides["validation_interval"] = args.validation_interval
    config = Config().copy_with(**overrides)
    require_cuda(config.device, verbose=True)
    summary = run_training(config)
    print(f"Training complete. Best checkpoint: {summary['best_checkpoint']}")
    return 0


def _load_checkpoint_config(checkpoint: str | Path, data_dir: Optional[str], device: str) -> Config:
    payload = load_checkpoint(checkpoint, map_location="cpu")
    stored = payload.get("config")
    config = Config.from_dict(stored) if isinstance(stored, dict) else Config()
    overrides: Dict[str, Any] = {"device": device}
    if data_dir is not None:
        overrides["data_dir"] = data_dir
    return config.copy_with(**overrides)


def _evaluate_command(args: argparse.Namespace) -> int:
    require_cuda(args.device, verbose=True)
    config = _load_checkpoint_config(args.checkpoint, args.data_dir, args.device)
    config = config.copy_with(results_dir=args.output_dir)
    instances, _ = load_dataset(config.test_path)
    if len(instances) != config.num_test_instances:
        raise ValueError(
            f"Test dataset has {len(instances)} instances; config expects "
            f"{config.num_test_instances}."
        )
    evaluate_checkpoint(
        config,
        args.checkpoint,
        instances,
        device=args.device,
        output_dir=args.output_dir,
        include_greedy=args.include_greedy,
    )
    return 0


def _train_seed(
    seed: int,
    *,
    base: Config,
    device: str,
    episodes: Optional[int],
    data_dir: Optional[str],
) -> Dict[str, Any]:
    checkpoint_dir = Path("checkpoints") / f"seed_{seed}"
    log_dir = Path("logs") / f"seed_{seed}"
    results_dir = Path("results") / f"seed_{seed}"
    overrides: Dict[str, Any] = {
        "training_seed": seed,
        "device": device,
        "checkpoint_dir": str(checkpoint_dir),
        "log_dir": str(log_dir),
        "results_dir": str(results_dir),
        "figures_dir": str(results_dir / "figures"),
    }
    if episodes is not None:
        overrides["episodes"] = episodes
    if data_dir is not None:
        overrides["data_dir"] = data_dir
    config = base.copy_with(**overrides)
    print("=" * 78)
    print(f"Formal seed {seed}: training")
    print("=" * 78)
    training_summary = run_training(config)
    checkpoint = training_summary["best_checkpoint"]
    instances, _ = load_dataset(config.test_path)
    if len(instances) != config.num_test_instances:
        raise ValueError(
            f"Test dataset has {len(instances)} instances; expected {config.num_test_instances}"
        )
    print(f"Formal seed {seed}: evaluating {len(instances)} held-out mazes")
    evaluation_summary = evaluate_checkpoint(
        config,
        checkpoint,
        instances,
        device=device,
        output_dir=results_dir,
        include_greedy=config.include_greedy_evaluation,
    )
    _cleanup_cuda()
    return {
        "seed": seed,
        "config": config,
        "training_summary": training_summary,
        "evaluation_summary": evaluation_summary,
        "results_dir": results_dir,
        "checkpoint": checkpoint,
    }


def _run_seeds(
    seeds: Sequence[int],
    *,
    base: Config,
    device: str,
    episodes: Optional[int],
    data_dir: Optional[str],
    create_figures: bool,
    base_results_dir: Path = Path("results"),
) -> Dict[str, Any]:
    if len(seeds) == 0:
        raise ValueError("At least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    _ensure_datasets(base)
    runs: List[Dict[str, Any]] = []
    for seed in seeds:
        runs.append(
            _train_seed(
                seed,
                base=base,
                device=device,
                episodes=episodes,
                data_dir=data_dir,
            )
        )

    combined_records: List[Dict[str, Any]] = []
    for run in runs:
        records = read_evaluation_csv(Path(run["results_dir"]) / "evaluation.csv")
        combined_records.extend(records)
    ensure_dir(base_results_dir)
    combined_csv = write_evaluation_csv(combined_records, base_results_dir / "evaluation.csv")
    seed_summaries = [run["evaluation_summary"] for run in runs]
    algorithms = sorted(
        {
            algorithm
            for summary in seed_summaries
            for algorithm in summary.get("algorithms", {})
        }
    )
    aggregates = seed_level_metric_aggregate(seed_summaries, algorithms)
    multi_summary = {
        "status": "completed",
        "created_at_utc": utc_now_iso(),
        "aggregation": "mean and sample standard deviation across random seeds",
        "seeds": [int(seed) for seed in seeds],
        "num_seeds": len(seeds),
        "episodes_per_seed": runs[0]["config"].episodes,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(torch.device(device)),
        "evaluation_csv": str(combined_csv),
        "per_seed": {
            str(run["seed"]): {
                "checkpoint": str(run["checkpoint"]),
                "results_dir": str(run["results_dir"]),
                "evaluation_summary": str(Path(run["results_dir"]) / "summary.json"),
                "training_summary": run["training_summary"],
                "algorithms": run["evaluation_summary"].get("algorithms", {}),
            }
            for run in runs
        },
        "algorithms": aggregates,
        "note": (
            "Ablation runs write to results/ablation/<name> and never overwrite "
            "this formal multi-seed summary."
        ),
    }
    summary_path = save_summary(multi_summary, base_results_dir / "summary.json")
    print(f"Formal multi-seed summary: {summary_path}")
    print(f"Combined test records: {combined_csv}")

    if create_figures:
        first_log_dir = Path("logs") / f"seed_{seeds[0]}"
        figures = visualize(
            training_log=first_log_dir / "training_log.csv",
            validation_log=first_log_dir / "validation_log.csv",
            evaluation_csv=combined_csv,
            figures_dir=base_results_dir / "figures",
        )
        print(
            f"Generated {len(figures)} figures in {base_results_dir / 'figures'} "
            f"(training curves use seed {seeds[0]}; algorithm comparisons aggregate all seeds)."
        )
    return multi_summary


def _multi_seed_command(args: argparse.Namespace) -> int:
    require_cuda(args.device, verbose=True)
    base = Config()
    if args.data_dir is not None:
        base = base.copy_with(data_dir=args.data_dir)
    seeds = tuple(args.seeds) if args.seeds else base.training_seeds
    _run_seeds(
        seeds,
        base=base,
        device=args.device,
        episodes=args.episodes,
        data_dir=args.data_dir,
        create_figures=not args.no_figures,
        base_results_dir=Path(args.results_dir),
    )
    return 0


def _ablation_config(base: Config, name: str) -> Config:
    if name == "no_solution_bounds":
        return base.copy_with(use_solution_bounds=False)
    if name == "uniform_sampling":
        return base.copy_with(use_instance_sampling=False)
    if name == "mlp_capacity_64":
        return base.copy_with(hidden_dims=(64, 64))
    if name == "mlp_capacity_128":
        return base.copy_with(hidden_dims=(128, 128))
    raise ValueError(f"Unknown ablation config: {name}")


def _run_one_ablation(
    name: str,
    *,
    base: Config,
    seed: int,
    episodes: Optional[int],
    device: str,
    data_dir: Optional[str],
) -> Dict[str, Any]:
    checkpoint_dir = Path("checkpoints") / "ablation" / name
    log_dir = Path("logs") / "ablation" / name
    output_dir = Path("results") / "ablation" / name
    overrides: Dict[str, Any] = {
        "training_seed": seed,
        "device": device,
        "checkpoint_dir": str(checkpoint_dir),
        "log_dir": str(log_dir),
        "results_dir": str(output_dir),
        "figures_dir": str(output_dir / "figures"),
    }
    if episodes is not None:
        overrides["episodes"] = episodes
    if data_dir is not None:
        overrides["data_dir"] = data_dir
    config = _ablation_config(base, name).copy_with(**overrides)
    print("=" * 78)
    print(f"Ablation '{name}': training seed {seed}")
    print("=" * 78)
    training_summary = run_training(config)
    instances, _ = load_dataset(config.test_path)
    evaluation_summary = evaluate_checkpoint(
        config,
        training_summary["best_checkpoint"],
        instances,
        device=device,
        output_dir=output_dir,
    )
    figures = visualize(
        training_log=log_dir / "training_log.csv",
        validation_log=log_dir / "validation_log.csv",
        evaluation_csv=output_dir / "evaluation.csv",
        figures_dir=output_dir / "figures",
    )
    result = {
        "status": "completed",
        "name": name,
        "seed": seed,
        "episodes": config.episodes,
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "training_summary": training_summary,
        "evaluation_summary": evaluation_summary,
        "figures": [str(path) for path in figures],
        "config": config.to_dict(),
    }
    save_json(result, output_dir / "ablation_summary.json")
    _cleanup_cuda()
    print(f"Ablation '{name}' complete: {output_dir}")
    return result


def _ablation_command(args: argparse.Namespace) -> int:
    require_cuda(args.device, verbose=True)
    base = Config()
    if args.data_dir is not None:
        base = base.copy_with(data_dir=args.data_dir)
    _ensure_datasets(base)
    if args.name == "mlp_capacity":
        names = ["mlp_capacity_64", "mlp_capacity_128"]
    else:
        names = [args.name]
    results = []
    for name in names:
        results.append(
            _run_one_ablation(
                name,
                base=base,
                seed=args.seed,
                episodes=args.episodes,
                device=args.device,
                data_dir=args.data_dir,
            )
        )
    comparison = {
        "status": "completed",
        "created_at_utc": utc_now_iso(),
        "ablation_group": args.name,
        "seed": args.seed,
        "runs": [
            {
                "name": result["name"],
                "output_dir": result["output_dir"],
                "evaluation_summary": result["evaluation_summary"],
                "config": {
                    "use_solution_bounds": result["config"]["use_solution_bounds"],
                    "use_instance_sampling": result["config"]["use_instance_sampling"],
                    "hidden_dims": result["config"]["hidden_dims"],
                },
            }
            for result in results
        ],
    }
    group_dir = Path("results") / "ablation"
    ensure_dir(group_dir)
    save_json(comparison, group_dir / f"{args.name}_comparison.json")
    print(f"Ablation group '{args.name}' finished.")
    return 0


def _visualize_command(args: argparse.Namespace) -> int:
    training_log = Path(args.training_log)
    validation_log = Path(args.validation_log)
    if not training_log.exists() or not validation_log.exists():
        for seed in (42, 43, 44):
            candidate_log = Path("logs") / f"seed_{seed}" / "training_log.csv"
            candidate_validation = Path("logs") / f"seed_{seed}" / "validation_log.csv"
            if candidate_log.exists() and candidate_validation.exists():
                training_log = candidate_log
                validation_log = candidate_validation
                print(f"Using training curves from seed {seed}.")
                break
    figures = visualize(
        training_log=training_log,
        validation_log=validation_log,
        evaluation_csv=args.evaluation_csv,
        figures_dir=args.figures_dir,
    )
    print(f"Generated {len(figures)} figures in {args.figures_dir}")
    return 0


def _all_command(args: argparse.Namespace) -> int:
    require_cuda(args.device, verbose=True)
    base = Config()
    if args.data_dir is not None:
        base = base.copy_with(data_dir=args.data_dir)
    if args.force_generate:
        generate_datasets(base, force=True)
    else:
        _ensure_datasets(base)
    seeds = tuple(args.seeds) if args.seeds else base.training_seeds
    _run_seeds(
        seeds,
        base=base,
        device=args.device,
        episodes=args.episodes,
        data_dir=args.data_dir,
        create_figures=not args.no_figures,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GSP reproduction experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)

    cuda = sub.add_parser("check-cuda", help="strict CUDA environment check")
    cuda.add_argument("--device", default="cuda:0")

    generate = sub.add_parser("generate", help="generate datasets")
    generate.add_argument("--force", action="store_true")
    generate.add_argument("--data-dir", default=None)

    train = sub.add_parser("train", help="run one training job")
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--episodes", type=int, default=None)
    train.add_argument("--validation-interval", type=int, default=None)
    train.add_argument("--data-dir", default=None)
    train.add_argument("--checkpoint-dir", default=None)
    train.add_argument("--log-dir", default=None)
    train.add_argument("--results-dir", default=None)

    multi = sub.add_parser("multi-seed", help="formal sequential three-seed experiment")
    multi.add_argument("--device", default="cuda:0")
    multi.add_argument("--seeds", type=int, nargs="+", default=None)
    multi.add_argument("--episodes", type=int, default=None)
    multi.add_argument("--data-dir", default=None)
    multi.add_argument("--results-dir", default="results")
    multi.add_argument("--no-figures", action="store_true")

    evaluate = sub.add_parser("evaluate", help="evaluate a checkpoint on the test set")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--data-dir", default=None)
    evaluate.add_argument("--output-dir", default="results")
    evaluate.add_argument("--include-greedy", action=argparse.BooleanOptionalAction, default=None)

    ablation = sub.add_parser("ablation", help="run a controlled ablation")
    ablation.add_argument("--name", choices=ABLATION_CHOICES, required=True)
    ablation.add_argument("--seed", type=int, default=42)
    ablation.add_argument("--episodes", type=int, default=None)
    ablation.add_argument("--device", default="cuda:0")
    ablation.add_argument("--data-dir", default=None)

    figs = sub.add_parser("visualize", help="generate all figures")
    figs.add_argument("--training-log", default="logs/training_log.csv")
    figs.add_argument("--validation-log", default="logs/validation_log.csv")
    figs.add_argument("--evaluation-csv", default="results/evaluation.csv")
    figs.add_argument("--figures-dir", default="results/figures")

    all_parser = sub.add_parser("all", help="generate data, train seeds sequentially, evaluate, plot")
    all_parser.add_argument("--device", default="cuda:0")
    all_parser.add_argument("--seeds", type=int, nargs="+", default=None)
    all_parser.add_argument("--episodes", type=int, default=None)
    all_parser.add_argument("--data-dir", default=None)
    all_parser.add_argument("--force-generate", action="store_true")
    all_parser.add_argument("--no-figures", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "check-cuda":
        require_cuda(args.device, verbose=True)
        return 0
    if args.command == "generate":
        config = Config()
        if args.data_dir is not None:
            config = config.copy_with(data_dir=args.data_dir)
        manifest = generate_datasets(config, force=args.force)
        print(json.dumps(manifest.get("splits", {}), indent=2, ensure_ascii=False))
        return 0
    if args.command == "train":
        return _single_train(args)
    if args.command == "multi-seed":
        return _multi_seed_command(args)
    if args.command == "evaluate":
        return _evaluate_command(args)
    if args.command == "ablation":
        return _ablation_command(args)
    if args.command == "visualize":
        return _visualize_command(args)
    if args.command == "all":
        return _all_command(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())