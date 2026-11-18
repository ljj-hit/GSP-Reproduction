"""Evaluation metrics and CSV/JSON helpers."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from utils import ensure_dir, save_json


PREFERRED_ALGORITHM_ORDER = (
    "BFS",
    "A* + Manhattan",
    "WA* + Manhattan",
    "Untrained-Q WA*",
    "GSP-WA*",
    "Greedy-Q",
)

EVALUATION_COLUMNS = (
    "algorithm",
    "instance_id",
    "difficulty",
    "bfs_length",
    "success",
    "expansions",
    "plan_length",
    "path_ratio",
    "search_time_seconds",
    "inference_time_seconds",
    "non_inference_time_seconds",
    "failure_reason",
    "seed",
)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            value = float(stripped)
        except ValueError:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _as_int(value: Any) -> Optional[int]:
    number = _as_float(value)
    if number is None:
        return None
    return int(number)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    """Mean and sample standard deviation (0 for a single value)."""
    cleaned = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not cleaned:
        return None, None
    if len(cleaned) == 1:
        return cleaned[0], 0.0
    return float(np.mean(cleaned)), float(np.std(cleaned, ddof=1))


def success_rate(results: Sequence[Any]) -> float:
    if len(results) == 0:
        raise ValueError("success_rate needs at least one result")
    return sum(1 for value in results if _as_bool(value)) / len(results)


def mean_of(values: Sequence[Any]) -> Optional[float]:
    mean, _ = mean_std([value for value in (_as_float(v) for v in values) if value is not None])
    return mean


def algorithm_sort_key(name: str) -> Tuple[int, str]:
    try:
        return PREFERRED_ALGORITHM_ORDER.index(name), name
    except ValueError:
        return len(PREFERRED_ALGORITHM_ORDER), name


def group_records_by_algorithm(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        algorithm = str(record["algorithm"])
        grouped.setdefault(algorithm, []).append(record)
    return dict(sorted(grouped.items(), key=lambda item: algorithm_sort_key(item[0])))


def summarize_algorithm(
    records: Sequence[Mapping[str, Any]],
    algorithm: str,
) -> Dict[str, Any]:
    """Summarise per-instance records for one algorithm.

    Expansions and times are reported over *all* attempts, so failures are not
    hidden.  Path-quality statistics are defined only for solved instances and
    therefore also report the solved count explicitly.
    """
    if not records:
        raise ValueError(f"No records for algorithm {algorithm}")
    successes = [_as_bool(record.get("success", False)) for record in records]
    solved_records = [record for record, solved in zip(records, successes) if solved]
    expansions_all = [_as_int(record.get("expansions")) for record in records]
    expansions_all_f = [value for value in expansions_all if value is not None]
    expansions_solved = [
        _as_int(record.get("expansions")) for record in solved_records
    ]
    expansions_solved_f = [value for value in expansions_solved if value is not None]
    plan_lengths = [_as_int(record.get("plan_length")) for record in solved_records]
    plan_lengths_f = [value for value in plan_lengths if value is not None]
    path_ratios = [
        _as_float(record.get("path_ratio")) for record in solved_records
    ]
    path_ratios_f = [value for value in path_ratios if value is not None]
    search_times = [_as_float(record.get("search_time_seconds")) for record in records]
    search_times_f = [value for value in search_times if value is not None]
    inference_times = [
        _as_float(record.get("inference_time_seconds")) for record in records
    ]
    inference_times_f = [value for value in inference_times if value is not None]
    non_inference_times = [
        _as_float(record.get("non_inference_time_seconds")) for record in records
    ]
    non_inference_times_f = [value for value in non_inference_times if value is not None]

    success_mean, success_std = mean_std([1.0 if value else 0.0 for value in successes])
    exp_all_mean, exp_all_std = mean_std(expansions_all_f)
    exp_solved_mean, exp_solved_std = mean_std(expansions_solved_f)
    plan_mean, plan_std = mean_std(plan_lengths_f)
    ratio_mean, ratio_std = mean_std(path_ratios_f)
    search_mean, search_std = mean_std(search_times_f)
    infer_mean, infer_std = mean_std(inference_times_f)
    control_mean, control_std = mean_std(non_inference_times_f)

    failure_reasons: Dict[str, int] = {}
    for record, solved in zip(records, successes):
        if solved:
            continue
        reason = str(record.get("failure_reason", "unknown") or "unknown")
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    return {
        "algorithm": algorithm,
        "num_instances": len(records),
        "num_solved": len(solved_records),
        "num_unresolved": len(records) - len(solved_records),
        "success_rate": success_mean,
        "success_rate_std": success_std,
        "mean_expansions_all": exp_all_mean,
        "std_expansions_all": exp_all_std,
        "mean_expansions_successful": exp_solved_mean,
        "std_expansions_successful": exp_solved_std,
        "mean_plan_length_successful": plan_mean,
        "std_plan_length_successful": plan_std,
        "mean_path_ratio_successful": ratio_mean,
        "std_path_ratio_successful": ratio_std,
        "mean_search_time_seconds": search_mean,
        "std_search_time_seconds": search_std,
        "mean_inference_time_seconds": infer_mean,
        "std_inference_time_seconds": infer_std,
        "mean_non_inference_time_seconds": control_mean,
        "std_non_inference_time_seconds": control_std,
        "failure_reasons": failure_reasons,
        "solved_for_path_metrics": len(solved_records),
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped = group_records_by_algorithm(records)
    return {
        algorithm: summarize_algorithm(group, algorithm)
        for algorithm, group in grouped.items()
    }


def write_evaluation_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVALUATION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in EVALUATION_COLUMNS})
    return path


def read_evaluation_csv(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    records: List[Dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: Dict[str, Any] = {
                "algorithm": row["algorithm"],
                "instance_id": _as_int(row.get("instance_id")),
                "difficulty": row.get("difficulty") or "unknown",
                "bfs_length": _as_int(row.get("bfs_length")),
                "success": _as_bool(row.get("success")),
                "expansions": _as_int(row.get("expansions")),
                "plan_length": _as_int(row.get("plan_length")),
                "path_ratio": _as_float(row.get("path_ratio")),
                "search_time_seconds": _as_float(row.get("search_time_seconds")),
                "inference_time_seconds": _as_float(row.get("inference_time_seconds")),
                "non_inference_time_seconds": _as_float(row.get("non_inference_time_seconds")),
                "failure_reason": row.get("failure_reason") or "",
            }
            seed = _as_int(row.get("seed"))
            if seed is not None:
                parsed["seed"] = seed
            records.append(parsed)
    return records


def save_summary(summary: Mapping[str, Any], path: str | Path) -> Path:
    return save_json(summary, path)


def seed_level_metric_aggregate(
    summaries: Sequence[Mapping[str, Any]],
    algorithms: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate single-run summaries across random seeds.

    Each algorithm entry reports the mean and sample standard deviation across
    seeds for every scalar metric.  Per-seed summaries are preserved elsewhere
    by the experiment runner.
    """
    summaries = list(summaries)
    algorithms = list(algorithms)
    aggregate: Dict[str, Dict[str, Any]] = {}
    for algorithm in algorithms:
        per_seed = [
            summary["algorithms"][algorithm]
            for summary in summaries
            if algorithm in summary.get("algorithms", {})
        ]
        if not per_seed:
            continue
        all_keys = set()
        for entry in per_seed:
            all_keys.update(entry.keys())
        metric_keys = [
            key
            for key in all_keys
            if isinstance(per_seed[0].get(key), (int, float))
            and key not in {"num_instances", "num_solved", "num_unresolved", "solved_for_path_metrics"}
        ]
        entry: Dict[str, Any] = {
            "num_seeds": len(per_seed),
            "metrics": {},
        }
        for key in sorted(metric_keys):
            values = [
                float(seed_entry[key])
                for seed_entry in per_seed
                if isinstance(seed_entry.get(key), (int, float))
            ]
            mean, std = mean_std(values)
            entry["metrics"][key] = {"mean": mean, "std": std, "values": values}
        aggregate[algorithm] = entry
    return aggregate