"""Generate standalone Matplotlib figures for the GSP reproduction.

All figures are saved separately under ``results/figures``.  The palette is a
validated categorical set; algorithm colours stay fixed across charts.  Bar
charts carry direct value labels and error bars so low-contrast hues are not
the sole carrier of information.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from metrics import algorithm_sort_key, read_evaluation_csv
from utils import ensure_dir, setup_logger


# Validated light-mode categorical palette (adjacent-pair safe).
SERIES_COLORS = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
)
ALGORITHM_COLORS = {
    "BFS": SERIES_COLORS[0],
    "A* + Manhattan": SERIES_COLORS[1],
    "WA* + Manhattan": SERIES_COLORS[2],
    "Untrained-Q WA*": SERIES_COLORS[3],
    "GSP-WA*": SERIES_COLORS[4],
    "Greedy-Q": SERIES_COLORS[5],
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 9.5,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.labelsize": 9.5,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def _finish_axes(ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_axisbelow(True)
    ax.grid(axis="x" if xgrid else "y", visible=True, color=GRID, linewidth=0.7)
    if xgrid and ygrid:
        ax.grid(True, color=GRID, linewidth=0.7)
    elif xgrid:
        ax.grid(axis="y", visible=False)
    else:
        ax.grid(axis="x", visible=False)


def _save(fig: plt.Figure, path: Path, logger: Any) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _parse_int(value: Any) -> Optional[int]:
    number = _parse_float(value)
    return None if number is None else int(number)


def plot_training_loss(training_log: Path, figures_dir: Path, logger: Any) -> Optional[Path]:
    if not training_log.is_file():
        logger.warning("Training log not found; skipping loss figure: %s", training_log)
        return None
    rows = _read_csv(training_log)
    episodes: List[int] = []
    losses: List[float] = []
    for row in rows:
        episode = _parse_int(row.get("episode"))
        loss = _parse_float(row.get("loss"))
        if episode is not None and loss is not None:
            episodes.append(episode)
            losses.append(max(loss, 1e-12))
    if not episodes:
        logger.warning("No numeric loss entries found; skipping loss figure")
        return None
    x = np.asarray(episodes, dtype=float)
    y = np.asarray(losses, dtype=float)
    window = min(25, max(2, len(y) // 20)) if len(y) >= 4 else 1
    if window > 1:
        kernel = np.ones(window) / window
        smooth = np.convolve(y, kernel, mode="valid")
        smooth_x = x[window - 1:]
    else:
        smooth = y
        smooth_x = x

    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    ax.plot(x, y, color=SERIES_COLORS[0], linewidth=1.0, alpha=0.28, label="Per-episode loss")
    ax.plot(
        smooth_x,
        smooth,
        color=SERIES_COLORS[0],
        linewidth=2.0,
        label=f"Moving average ({window} episodes)",
    )
    _finish_axes(ax)
    ax.set_title("Training loss over episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    if len(smooth_x):
        ax.annotate(
            f"{smooth[-1]:.4f}",
            xy=(smooth_x[-1], smooth[-1]),
            xytext=(5, 4),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=8.5,
        )
    path = figures_dir / "training_loss.png"
    _save(fig, path, logger)
    return path


def _load_validation_rows(path: Path) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for row in _read_csv(path):
        episode = _parse_int(row.get("episode"))
        if episode is None:
            continue
        parsed.append(
            {
                "episode": episode,
                "success_rate": _parse_float(row.get("success_rate")),
                "mean_expansions_all": _parse_float(row.get("mean_expansions_all")),
                "mean_expansions_solved": _parse_float(row.get("mean_expansions_solved")),
                "mean_plan_length_solved": _parse_float(row.get("mean_plan_length_solved")),
                "num_solved": _parse_int(row.get("num_solved")),
                "num_instances": _parse_int(row.get("num_instances")),
            }
        )
    return parsed


def plot_validation_success(validation_log: Path, figures_dir: Path, logger: Any) -> Optional[Path]:
    if not validation_log.is_file():
        logger.warning("Validation log not found; skipping success curve")
        return None
    rows = [row for row in _load_validation_rows(validation_log) if row["success_rate"] is not None]
    if not rows:
        return None
    x = np.asarray([row["episode"] for row in rows], dtype=float)
    y = np.asarray([row["success_rate"] * 100.0 for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    ax.plot(
        x,
        y,
        color=SERIES_COLORS[0],
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        label="Validation success rate",
    )
    _finish_axes(ax)
    ax.set_title("Validation success rate during training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    ax.annotate(
        f"{y[-1]:.1f}%",
        xy=(x[-1], y[-1]),
        xytext=(5, 4),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=8.5,
    )
    path = figures_dir / "validation_success_rate.png"
    _save(fig, path, logger)
    return path


def plot_validation_expansions(validation_log: Path, figures_dir: Path, logger: Any) -> Optional[Path]:
    if not validation_log.is_file():
        logger.warning("Validation log not found; skipping expansion curve")
        return None
    rows = [row for row in _load_validation_rows(validation_log) if row["mean_expansions_all"] is not None]
    if not rows:
        return None
    x = np.asarray([row["episode"] for row in rows], dtype=float)
    all_exp = np.asarray([row["mean_expansions_all"] for row in rows], dtype=float)
    solved_exp = np.asarray(
        [np.nan if row["mean_expansions_solved"] is None else row["mean_expansions_solved"] for row in rows],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    ax.plot(
        x,
        all_exp,
        color=SERIES_COLORS[0],
        linewidth=2.0,
        marker="o",
        markersize=4.5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        label="All validation instances",
    )
    if np.isfinite(solved_exp).any():
        ax.plot(
            x,
            solved_exp,
            color=SERIES_COLORS[1],
            linewidth=2.0,
            marker="s",
            markersize=4.0,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            label="Solved validation instances",
        )
    _finish_axes(ax)
    ax.set_title("Validation node expansions during training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean expansions per maze")
    ax.legend(loc="best")
    path = figures_dir / "validation_expansions.png"
    _save(fig, path, logger)
    return path


def _metric_values(
    records: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    only_successful: bool = False,
    scale: float = 1.0,
) -> Tuple[Optional[float], Optional[float], str, int]:
    """Compute mean/std for one metric for one algorithm.

    With multiple random seeds, the unit of aggregation is the seed (mean per
    seed, then std across seeds).  With one seed, units are test instances.
    """
    selected = [
        record
        for record in records
        if (not only_successful or bool(record.get("success", False)))
    ]
    if not selected:
        return None, None, "no observations", 0
    seeds = sorted({int(record["seed"]) for record in selected if record.get("seed") is not None})
    if len(seeds) > 1:
        seed_means: List[float] = []
        for seed in seeds:
            values = []
            for record in selected:
                if record.get("seed") is None or int(record["seed"]) != seed:
                    continue
                value = _parse_float(record.get(metric))
                if value is not None:
                    values.append(value * scale)
            if values:
                seed_means.append(float(np.mean(values)))
        if not seed_means:
            return None, None, "no observations", 0
        if len(seed_means) == 1:
            return seed_means[0], 0.0, "across random seeds", 1
        return (
            float(np.mean(seed_means)),
            float(np.std(seed_means, ddof=1)),
            "across random seeds",
            len(seed_means),
        )
    values = [
        _parse_float(record.get(metric)) * scale
        for record in selected
        if _parse_float(record.get(metric)) is not None
    ]
    if not values:
        return None, None, "no observations", 0
    if len(values) == 1:
        return values[0], 0.0, "across test instances", 1
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1)),
        "across test instances",
        len(values),
    )


def _bar_chart(
    records: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    title: str,
    xlabel: str,
    output_name: str,
    figures_dir: Path,
    logger: Any,
    scale: float = 1.0,
    value_format: str = ".2f",
    success_metric: bool = False,
    only_successful: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    note: Optional[str] = None,
) -> Optional[Path]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["algorithm"]), []).append(record)
    algorithms = sorted(grouped, key=algorithm_sort_key)
    if not algorithms:
        logger.warning("No records available for %s", title)
        return None

    means: List[float] = []
    errors: List[float] = []
    labels: List[str] = []
    kept_algorithms: List[str] = []
    error_source = "across test instances"
    for algorithm in algorithms:
        if success_metric:
            values_with_seed = [
                (
                    record.get("seed"),
                    1.0 if bool(record.get("success", False)) else 0.0,
                )
                for record in grouped[algorithm]
            ]
            seeds = sorted({int(seed) for seed, _ in values_with_seed if seed is not None})
            if len(seeds) > 1:
                seed_values = [
                    np.mean([value for seed, value in values_with_seed if int(seed) == seed_id])
                    for seed_id in seeds
                ]
                mean = float(np.mean(seed_values)) * scale
                std = (
                    float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else 0.0
                ) * scale
                error_source = "across random seeds"
            else:
                values = np.asarray([value for _, value in values_with_seed], dtype=float) * scale
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                error_source = "across test instances"
            if scale == 100.0:
                label = f"{mean:.1f}%"
            else:
                label = f"{mean:.4f}"
        else:
            mean, std, source, count = _metric_values(
                grouped[algorithm],
                metric=metric,
                only_successful=only_successful,
                scale=scale,
            )
            error_source = source
            if mean is None or std is None or count == 0:
                continue
            label = f"{mean:{value_format}}"
        means.append(mean)
        errors.append(std if std is not None else 0.0)
        labels.append(label)
        kept_algorithms.append(algorithm)

    if not kept_algorithms:
        logger.warning("No observations for %s; skipping figure", title)
        return None

    positions = np.arange(len(kept_algorithms))
    colors = [ALGORITHM_COLORS.get(name, SERIES_COLORS[i % len(SERIES_COLORS)]) for i, name in enumerate(kept_algorithms)]
    fig_height = max(3.8, 0.72 * len(kept_algorithms) + 2.0)
    fig, ax = plt.subplots(figsize=(8.4, fig_height), constrained_layout=True)
    bars = ax.barh(
        positions,
        means,
        height=0.46,
        color=colors,
        edgecolor="none",
        xerr=errors,
        error_kw={
            "ecolor": INK_SECONDARY,
            "elinewidth": 1.0,
            "capsize": 3,
            "capthick": 1.0,
        },
        zorder=2,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(kept_algorithms, color=INK)
    ax.invert_yaxis()
    _finish_axes(ax, xgrid=True, ygrid=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_axisbelow(True)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        upper_candidates = [mean + error for mean, error in zip(means, errors) if mean is not None]
        upper = max(upper_candidates) if upper_candidates else 1.0
        lower = min(0.0, min(means))
        ax.set_xlim(lower, max(upper * 1.20, upper + 0.02))
    x_low, x_high = ax.get_xlim()
    label_offset = (x_high - x_low) * 0.012
    for bar, mean, error, label in zip(bars, means, errors, labels):
        # Value at the bar tip; the surface-coloured box keeps it legible
        # where the whisker passes behind the label.
        ax.text(
            bar.get_width() + label_offset,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            color=INK,
            fontsize=8.5,
            fontweight="semibold",
            clip_on=False,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0},
        )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=INK_SECONDARY,
            linewidth=1.0,
            marker="|",
            markersize=7,
            label=f"Mean ± std ({error_source})",
        )
    ]
    ax.legend(handles=legend_handles, loc="lower right")
    if note:
        fig.text(
            0.01,
            -0.035,
            note,
            color=INK_MUTED,
            fontsize=8.2,
            ha="left",
            va="top",
            in_layout=True,
        )
    path = figures_dir / output_name
    _save(fig, path, logger)
    return path


def plot_algorithm_comparisons(
    evaluation_csv: Path,
    figures_dir: Path,
    logger: Any,
) -> List[Path]:
    if not evaluation_csv.is_file():
        logger.warning("Evaluation CSV not found; skipping comparison figures: %s", evaluation_csv)
        return []
    records = read_evaluation_csv(evaluation_csv)
    if not records:
        logger.warning("Evaluation CSV is empty; skipping comparison figures")
        return []
    created: List[Path] = []
    common = dict(records=records, figures_dir=figures_dir, logger=logger)
    for kwargs in (
        dict(
            metric="success",
            success_metric=True,
            title="Test success rate by algorithm",
            xlabel="Success rate (%)",
            output_name="test_success_rate.png",
            scale=100.0,
            xlim=(0, 120),
            note="Bars: mean; whiskers: standard deviation. Failures count as 0%.",
        ),
        dict(
            metric="expansions",
            title="Test node expansions by algorithm",
            xlabel="Mean state/action expansions per maze",
            output_name="test_expansions.png",
            value_format=".1f",
            note="Expansions include failed attempts and are averaged over all test mazes.",
        ),
        dict(
            metric="path_ratio",
            only_successful=True,
            title="Test path quality by algorithm",
            xlabel="Path length / BFS shortest length",
            output_name="test_path_quality.png",
            value_format=".3f",
            note="Path ratio is defined on solved mazes only; success rates are shown separately.",
        ),
        dict(
            metric="search_time_seconds",
            title="End-to-end search time by algorithm",
            xlabel="Wall-clock time per maze (ms)",
            output_name="test_search_time.png",
            scale=1000.0,
            value_format=".2f",
            note="Includes data-structure control and, for learned methods, Q-network inference.",
        ),
        dict(
            metric="inference_time_seconds",
            title="Q-network inference time by algorithm",
            xlabel="Network inference time per maze (ms)",
            output_name="test_inference_time.png",
            scale=1000.0,
            value_format=".2f",
            note="Search-time component attributable to Q-network calls; search logic runs on CPU.",
        ),
    ):
        path = _bar_chart(**common, **kwargs)
        if path is not None:
            created.append(path)
    return created


def visualize(
    *,
    training_log: str | Path = "logs/training_log.csv",
    validation_log: str | Path = "logs/validation_log.csv",
    evaluation_csv: str | Path = "results/evaluation.csv",
    figures_dir: str | Path = "results/figures",
) -> List[Path]:
    configure_style()
    figures_dir = ensure_dir(figures_dir)
    logger = setup_logger("gsp.visualize", figures_dir / "visualization.log")
    created: List[Path] = []
    for function, source in (
        (plot_training_loss, Path(training_log)),
        (plot_validation_success, Path(validation_log)),
        (plot_validation_expansions, Path(validation_log)),
    ):
        path = function(source, figures_dir, logger)
        if path is not None:
            created.append(path)
    created.extend(plot_algorithm_comparisons(Path(evaluation_csv), figures_dir, logger))
    logger.info("Generated %d figures in %s", len(created), figures_dir)
    return created


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate GSP experiment figures")
    parser.add_argument("--training-log", type=str, default="logs/training_log.csv")
    parser.add_argument("--validation-log", type=str, default="logs/validation_log.csv")
    parser.add_argument("--evaluation-csv", type=str, default="results/evaluation.csv")
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    args = parser.parse_args(argv)
    figures = visualize(
        training_log=args.training_log,
        validation_log=args.validation_log,
        evaluation_csv=args.evaluation_csv,
        figures_dir=args.figures_dir,
    )
    if not figures:
        print("No figures were generated because no matching result files were found.")
        return 1
    print(f"Generated {len(figures)} figures in {args.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())