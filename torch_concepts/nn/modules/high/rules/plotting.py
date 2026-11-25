"""Matplotlib figures for monitoring temporal CMR training.

Figures are built on :class:`matplotlib.figure.Figure` with the Agg canvas, so
no GUI backend or global ``pyplot`` state is involved; they are safe to create
inside training loops. ``figure_to_array`` turns a figure into an RGB array
that any Lightning logger (W&B, TensorBoard) accepts.
"""

from typing import Optional, Sequence

import numpy as np

# Validated categorical order (identity), single-hue sequential ramp
# (magnitude), and recessive ink for text, axes and grid.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK = "#2b2b29"
MUTED = "#73726c"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _require_matplotlib():
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Plotting requires matplotlib: pip install matplotlib"
        ) from exc
    return Figure, FigureCanvasAgg


def _new_figure(width: float, height: float):
    Figure, FigureCanvasAgg = _require_matplotlib()
    figure = Figure(figsize=(width, height), dpi=110, facecolor=SURFACE)
    FigureCanvasAgg(figure)
    return figure


def _style(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7, length=2)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def _color(index: int) -> str:
    # Concepts beyond the validated slots fall back to muted ink rather than a
    # generated hue; the legend still names them.
    return SERIES[index] if index < len(SERIES) else MUTED


def figure_to_array(figure) -> np.ndarray:
    """Render a figure to an ``(height, width, 3)`` uint8 array."""
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()


def plot_concept_traces(
    inputs: np.ndarray,
    predicted: np.ndarray,
    true: np.ndarray,
    concept_names: Sequence[str],
    task_probabilities: Optional[np.ndarray] = None,
    task_targets: Optional[np.ndarray] = None,
    task_names: Optional[Sequence[str]] = None,
    title: str = "",
    column_titles: Optional[Sequence[str]] = None,
):
    """Input signal and predicted vs. true concept traces for a few sequences.

    One column per sequence; the first row is the raw input, then one row per
    concept with the ground truth as a light filled step and the prediction as
    a line in the same hue.

    Args:
        inputs: ``(n, time, features)``.
        predicted, true: ``(n, time, concepts)``.
        task_probabilities, task_targets: ``(n, tasks)``, shown in the titles.
        column_titles: Optional first title line per sequence (default
            ``"sequence i"``), e.g. why it was selected.
    """
    n_examples, time, _ = predicted.shape
    n_concepts = len(concept_names)
    # Reserve room above the plots for the figure title + legend row and each
    # column's header (one line, plus one per task when probabilities are shown).
    header_lines = 1 + (len(task_names) if task_probabilities is not None and task_names else 0)
    top_inches = 0.35 + 0.15 * header_lines
    height = top_inches + 0.5 + 0.9 * (n_concepts + 1)
    figure = _new_figure(3.6 * n_examples, height)
    axes = figure.subplots(
        n_concepts + 1, n_examples, sharex=True, squeeze=False,
        gridspec_kw={"hspace": 0.25, "wspace": 0.12},
    )
    figure.subplots_adjust(top=1 - top_inches / height, bottom=0.45 / height)
    steps = np.arange(time)
    for column in range(n_examples):
        ax = axes[0, column]
        _style(ax)
        for feature in range(inputs.shape[-1]):
            ax.plot(steps, inputs[column, :, feature], color=INK, linewidth=0.8)
        header = column_titles[column] if column_titles else f"sequence {column}"
        if task_probabilities is not None and task_names is not None:
            parts = []
            for task, name in enumerate(task_names):
                target = "" if task_targets is None else f"/{int(task_targets[column, task])}"
                parts.append(f"{name}: {task_probabilities[column, task]:.2f}{target}")
            header += "\n" + "\n".join(parts)
        ax.set_title(header, fontsize=7, color=INK, loc="left")
        if column == 0:
            ax.set_ylabel("input", fontsize=7, color=MUTED)

        for concept, name in enumerate(concept_names):
            ax = axes[concept + 1, column]
            _style(ax)
            color = _color(concept)
            ax.fill_between(
                steps, true[column, :, concept], step="mid", color=color,
                alpha=0.22, linewidth=0, label="true",
            )
            ax.plot(
                steps, predicted[column, :, concept], color=color,
                linewidth=1.2, label="predicted",
            )
            if column == 0:
                ax.set_ylabel(name, fontsize=7, color=INK, rotation=0, ha="right", va="center")
    axes[-1, 0].set_xlabel("time step", fontsize=7, color=MUTED)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(
        handles, ["true (filled)", "predicted (line)"], loc="upper right",
        bbox_to_anchor=(0.99, 1 - 0.04 / height),
        fontsize=7, frameon=False, ncol=2,
    )
    if title:
        figure.suptitle(
            title, fontsize=9, color=INK, x=0.01, y=1 - 0.06 / height, ha="left", va="top"
        )
    return figure


def plot_rule_bank(
    antecedent_weights: np.ndarray,
    consequent_weights: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    max_horizon: float,
    task_names: Sequence[str],
    concept_names: Sequence[str],
    selection: Optional[np.ndarray] = None,
    title: str = "",
):
    """Snapshot of a learned SNF bank ``G(A -> F_[start,end] B)``.

    Left: antecedent and consequent concept weights (softmax) per rule. Right:
    each rule's interval; bar opacity is its mean selection weight.

    Args:
        antecedent_weights, consequent_weights: ``(tasks, rules, concepts)``.
        starts, ends: ``(tasks, rules)`` in time steps.
        selection: optional ``(tasks, rules)`` mean selection weights.
    """
    from matplotlib.colors import LinearSegmentedColormap

    n_tasks, n_rules, n_concepts = antecedent_weights.shape
    rows = [f"{task} · r{rule}" for task in task_names for rule in range(n_rules)]
    figure = _new_figure(10.5, 0.9 + 0.32 * len(rows))
    axes = figure.subplots(
        1, 3, gridspec_kw={"width_ratios": [n_concepts, n_concepts, 8], "wspace": 0.08}
    )
    cmap = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL)

    for ax, weights, label in (
        (axes[0], antecedent_weights, "antecedent A"),
        (axes[1], consequent_weights, "consequent B"),
    ):
        matrix = weights.reshape(-1, n_concepts)
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for (row, column), value in np.ndenumerate(matrix):
            ax.text(
                column, row, f"{value:.2f}", ha="center", va="center", fontsize=6,
                color="white" if value > 0.55 else INK,
            )
        ax.set_xticks(range(n_concepts), concept_names, rotation=35, ha="right", fontsize=7, color=INK)
        ax.set_title(label, fontsize=8, color=INK)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_yticks(range(len(rows)), rows, fontsize=7, color=INK)
    axes[1].set_yticks([])

    ax = axes[2]
    _style(ax, grid_axis="x")
    flat_start, flat_end = starts.reshape(-1), ends.reshape(-1)
    alphas = (
        np.full(len(rows), 0.9)
        if selection is None
        else 0.25 + 0.75 * selection.reshape(-1)
    )
    for row in range(len(rows)):
        ax.barh(
            row, flat_end[row] - flat_start[row], left=flat_start[row], height=0.55,
            color=SERIES[0], alpha=float(alphas[row]),
        )
        note = f"[{flat_start[row]:.0f}, {flat_end[row]:.0f}]"
        if selection is not None:
            note += f"  sel {selection.reshape(-1)[row]:.2f}"
        ax.text(max_horizon * 1.01, row, note, va="center", fontsize=6, color=MUTED)
    ax.set_xlim(0, max_horizon * 1.25)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xlabel("Eventually interval (time steps)", fontsize=7, color=MUTED)
    ax.set_title("interval (opacity = selection)", fontsize=8, color=INK)
    if title:
        figure.suptitle(title, fontsize=9, color=INK, x=0.01, ha="left")
    figure.subplots_adjust(left=0.2, right=0.98, bottom=0.25 if len(rows) < 8 else 0.12)
    return figure


def plot_rule_history(
    epochs: Sequence[int],
    starts: np.ndarray,
    ends: np.ndarray,
    task_names: Sequence[str],
    robustness_scale: Optional[Sequence[float]] = None,
):
    """How each rule's interval moved over training.

    Args:
        epochs: ``(epochs,)``.
        starts, ends: ``(epochs, tasks, rules)``.
        robustness_scale: optional ``(epochs,)``; drawn in its own panel.
    """
    n_tasks, n_rules = starts.shape[1:]
    panels = n_tasks + (robustness_scale is not None)
    figure = _new_figure(3.4 * panels, 2.8)
    axes = np.atleast_1d(figure.subplots(1, panels, squeeze=False)[0])
    epochs = np.asarray(epochs)
    for task, name in enumerate(task_names):
        ax = axes[task]
        _style(ax)
        for rule in range(n_rules):
            color = _color(rule)
            ax.fill_between(
                epochs, starts[:, task, rule], ends[:, task, rule],
                color=color, alpha=0.18, linewidth=0,
            )
            ax.plot(epochs, starts[:, task, rule], color=color, linewidth=1.2, label=f"r{rule}")
            ax.plot(epochs, ends[:, task, rule], color=color, linewidth=1.2)
        ax.set_title(name, fontsize=8, color=INK, loc="left")
        ax.set_xlabel("epoch", fontsize=7, color=MUTED)
        if task == 0:
            ax.set_ylabel("interval [start, end]", fontsize=7, color=MUTED)
        if n_rules > 1:
            ax.legend(fontsize=6, frameon=False, loc="best")
    if robustness_scale is not None:
        ax = axes[-1]
        _style(ax)
        ax.plot(epochs, robustness_scale, color=INK, linewidth=1.2)
        ax.set_title("robustness scale", fontsize=8, color=INK, loc="left")
        ax.set_xlabel("epoch", fontsize=7, color=MUTED)
    figure.tight_layout()
    return figure
