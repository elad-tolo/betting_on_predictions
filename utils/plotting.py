from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

BASELINES_LABEL = "Baselines w/o Ours"
BASELINES_WITH_IMPUTED_LABEL = "Baselines with Ours"

KEY_TO_PLOT_CONFIG = {
    "imputed_with_baselines": (BASELINES_WITH_IMPUTED_LABEL, "tab:blue", "o"),
    "imputed_with_baselines_positive_lambda": (
        f"{BASELINES_WITH_IMPUTED_LABEL} with one-sided PPI",
        "tab:blue",
        "o",
    ),
    "baselines": (BASELINES_LABEL, "tab:orange", "s"),
    "baselines_positive_lambda": (
        f"{BASELINES_LABEL} with one-sided PPI",
        "tab:orange",
        "s",
    ),
    "NB_max_predictions_only": ("Only Ours", "tab:red", "X"),
    "NB_predictions_only": ("Only Ours", "tab:red", "X"),
    "TabPFN_predictions_only": ("Only Ours", "tab:red", "X"),
    "lrt_y": ("$e^{Y}_{LR}$", "tab:green", "v"),
    "lrt_x": ("$e^{X}_{LR}$", "tab:purple", "<"),
    "cond_lrt": (r"$e^{Y\mid X}_{LR}$", "tab:cyan", "D"),
    "PPI": ("PPI", "tab:olive", "^"),
    "PPI_positive_lambda": ("One-Sided PPI", "cyan", "v"),
    "NB_max_prod_predictions_only": ("Predictions Prod", "cyan", "p"),
    "PPI_no_unlabeled": ("PPI without unlabeled", "tab:blue", "X"),
    "M=2": ("M=2", "tab:blue", "o"),
    "M=5": ("M=5", "tab:orange", "s"),
    "M=16": ("M=16", "tab:green", "D"),
    "M=32": ("M=32", "tab:red", "^"),
    "M=128": ("M=128", "tab:purple", "v"),
    "alpha=0.05": (r"$\alpha=0.05$", "black", "p"),
}


def plot_multiple_curves(
    y_lists,
    x,
    y_err,
    keys,
    xlabel="X",
    ylabel="Y",
    title=None,
    show=True,
    save_path=None,
    ylim=(0.0, 1.0),
    show_legend=True,
    font_size=25,
    legend_font_size=20,
):
    """Plot multiple series sharing the same x-axis.

    Parameters
    ----------
    y_lists : list of array-like
        One series of y-values per curve.
    x : array-like
        X-axis values shared by all series.
    y_err : list of array-like or None
        Per-series standard errors; rendered as a shaded band.
    keys : list of str
        Lookup keys into ``KEY_TO_PLOT_CONFIG`` for label, color, marker.
    font_size : int
        Font size for axis labels, title, and tick labels.
    legend_font_size : int
        Font size for the legend (when ``show_legend=True``).
    """

    fig, ax = plt.subplots()

    for i, y in enumerate(y_lists):
        key = keys[i]
        lab, color, marker = KEY_TO_PLOT_CONFIG[key]
        ax.plot(
            x,
            y,
            marker=marker,
            linewidth=1,
            label=lab,
            markersize=6,
            color=color,
        )
        if y_err is not None:
            err = y_err[i]
            y_upper = [y_val + err_val for y_val, err_val in zip(y, err)]
            y_lower = [y_val - err_val for y_val, err_val in zip(y, err)]
            ax.fill_between(x, y_lower, y_upper, alpha=0.2, color=color)

    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_ylim(ylim)
    if title:
        ax.set_title(title, fontsize=font_size)
    ax.grid(True, linestyle="--", alpha=0.5)
    if show_legend:
        ax.legend(fontsize=legend_font_size)
    ax.tick_params(axis="both", which="major", labelsize=font_size)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()

    return fig, ax


def save_legend(axes, save_path=None, font_size=20):
    lines = []
    labels = []
    seen = set()

    for ax in axes:
        ax_lines, ax_labels = ax.get_legend_handles_labels()
        for line, label in zip(ax_lines, ax_labels):
            if label not in seen:
                seen.add(label)
                lines.append(line)
                labels.append(label)

    if not lines:
        print("No legend entries found.")
        return

    # Create a new figure for the legend
    # Estimate width based on labels length, this is a heuristic
    fig_legend = plt.figure(figsize=(len(labels) * 3, 1.5))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis("off")

    # Create the legend on this separate figure
    legend = ax_legend.legend(
        lines,
        labels,
        loc="center",
        ncol=len(labels),
        frameon=True,
        fontsize=font_size,
        edgecolor="black",
        fancybox=False,
    )

    fig_legend.canvas.draw()
    bbox = (
        legend.get_window_extent()
        .transformed(fig_legend.dpi_scale_trans.inverted())
        .expanded(1.1, 1.1)
    )

    if save_path:
        fig_legend.savefig(save_path, bbox_inches=bbox)
        print(f"Saved legend to {save_path}")
    plt.show()


def get_power_vs_steps(statistic_to_rejections, num_steps=None):
    num_steps_to_plot = 24
    max_step = num_steps if num_steps is not None else 500
    step_size = max_step // num_steps_to_plot
    steps = list(range(1, max_step, step_size))
    if steps[-1] != max_step:
        steps.append(max_step)
    statistic_to_power_over_time = defaultdict(list)
    statistic_to_power_stderr_over_time = defaultdict(list)
    for step in steps:
        for statistic, first_rejection_steps in statistic_to_rejections.items():
            tmp = np.array(first_rejection_steps, dtype=float)
            power = np.mean(tmp <= step)
            stderr = np.std((tmp <= step).astype(float), ddof=1) / np.sqrt(
                len(first_rejection_steps)
            )
            statistic_to_power_over_time[statistic].append(power)
            statistic_to_power_stderr_over_time[statistic].append(stderr)

    return steps, statistic_to_power_over_time, statistic_to_power_stderr_over_time


def plot_power_vs_steps_ppi_compared_labeled_data(
    statistic_to_rejections, title=None, save_path=None
):
    steps, statistic_to_power_over_time, statistic_to_power_stderr_over_time = (
        get_power_vs_steps(statistic_to_rejections)
    )

    key_to_label = {
        "PPI": "PPI",
        "PPI_no_unlabeled": r"PPI $\epsilon=0$",
    }

    y_lists = [statistic_to_power_over_time[key] for key in key_to_label]
    y_lists_err = [statistic_to_power_stderr_over_time[key] for key in key_to_label]

    return plot_multiple_curves(
        y_lists,
        steps,
        y_err=y_lists_err,
        keys=list(key_to_label.keys()),
        xlabel="# steps",
        ylabel="Power",
        title=title,
        show=True,
        save_path=save_path,
        show_legend=False,
    )


def plot_power_vs_steps_one_sided_ppi(
    statistic_to_rejections, title=None, save_path=None
):
    steps, statistic_to_power_over_time, statistic_to_power_stderr_over_time = (
        get_power_vs_steps(statistic_to_rejections)
    )

    pred_name = (
        "NB_max_predictions_only"
        if "NB_max_predictions_only" in statistic_to_power_over_time
        else "NB_predictions_only"
    )
    key_to_label = {
        "imputed_with_baselines_positive_lambda": "Baselines + Ours",
        "baselines_positive_lambda": "Baselines",
        "PPI": "PPI",
        "PPI_positive_lambda": "One Sided PPI",
        pred_name: "Our Method",
    }

    y_lists = [statistic_to_power_over_time[key] for key in key_to_label]
    y_lists_err = [statistic_to_power_stderr_over_time[key] for key in key_to_label]

    return plot_multiple_curves(
        y_lists,
        steps,
        y_err=y_lists_err,
        keys=list(key_to_label.keys()),
        xlabel="# steps",
        ylabel="Power",
        title=title,
        show=True,
        save_path=save_path,
        show_legend=False,
    )


def plot_power_vs_steps_paper(
    statistic_to_rejections,
    title=None,
    save_path=None,
    num_steps=None,
    show_legend=False,
    font_size=25,
    legend_font_size=20,
):
    steps, statistic_to_power_over_time, statistic_to_power_stderr_over_time = (
        get_power_vs_steps(statistic_to_rejections, num_steps=num_steps)
    )

    if "NB_max_predictions_only" in statistic_to_power_over_time:
        ml_martingale = "NB_max_predictions_only"
        baselines = ["lrt_y", "lrt_x"]
    elif "NB_predictions_only" in statistic_to_power_over_time:
        ml_martingale = "NB_predictions_only"
        baselines = ["lrt_y", "cond_lrt"]
    else:
        ml_martingale = "TabPFN_predictions_only"
        baselines = ["lrt_y"]

    statistics_to_plot = [
        "imputed_with_baselines",
        "baselines",
        ml_martingale,
    ] + baselines
    y_lists = [statistic_to_power_over_time[key] for key in statistics_to_plot]
    y_lists_err = [
        statistic_to_power_stderr_over_time[key] for key in statistics_to_plot
    ]

    return plot_multiple_curves(
        y_lists,
        steps,
        y_err=y_lists_err,
        keys=statistics_to_plot,
        xlabel="Step",
        ylabel="Power",
        title=title,
        show=True,
        save_path=save_path,
        show_legend=show_legend,
        font_size=font_size,
        legend_font_size=legend_font_size,
    )


def plot_power_vs_steps_validity(statistic_to_rejections, title=None, save_path=None):
    steps, statistic_to_power_over_time, statistic_to_power_stderr_over_time = (
        get_power_vs_steps(statistic_to_rejections)
    )

    key_to_label = {
        "label_shift_high": "Label Shift N=100",
        "label_shift_low": "Label Shift N=10",
        "concept_shift": "Concept Shift",
    }

    valid_keys = [k for k in key_to_label if k in statistic_to_power_over_time]

    y_lists = [statistic_to_power_over_time[key] for key in valid_keys]
    y_lists_err = [statistic_to_power_stderr_over_time[key] for key in valid_keys]

    # Add horizontal line for alpha=0.05
    y_lists.append([0.05] * len(steps))
    y_lists_err.append([0.0] * len(steps))
    valid_keys.append("alpha=0.05")

    return plot_multiple_curves(
        y_lists,
        steps,
        y_err=y_lists_err,
        keys=valid_keys,
        xlabel="Step",
        ylabel="Power",
        title=title,
        show=True,
        save_path=save_path,
        ylim=(0, 0.1),
    )


def plot_power_vs_steps_baselines(
    statistic_to_rejections, title=None, save_path=None, num_steps=None
):
    steps, statistic_to_power_over_time, statistic_to_power_stderr_over_time = (
        get_power_vs_steps(statistic_to_rejections, num_steps=num_steps)
    )

    key_to_label = {
        "lrt_y": "$e^{Y}_{LR}$",
        "lrt_x": "$e^{X}_{LR}$",
        "cond_lrt": r"$e^{Y\mid X}_{LR}$",
        "PPI": "PPI",
    }

    y_lists = [
        statistic_to_power_over_time[key]
        for key in key_to_label
        if key in statistic_to_power_over_time
    ]
    y_lists_err = [
        statistic_to_power_stderr_over_time[key]
        for key in key_to_label
        if key in statistic_to_power_over_time
    ]

    return plot_multiple_curves(
        y_lists,
        steps,
        y_err=y_lists_err,
        keys=[key for key in key_to_label if key in statistic_to_power_over_time],
        xlabel="Step",
        ylabel="Power",
        title=title,
        show=True,
        save_path=save_path,
        show_legend=False,
    )


def plot_hyperparameter_tuning(statistic_to_rejections, title=None, save_path=None):
    steps, statistic_to_power_over_time, statistic_to_power_stderr_over_time = (
        get_power_vs_steps(statistic_to_rejections)
    )

    y_lists = [statistic_to_power_over_time[key] for key in statistic_to_rejections]
    y_lists_err = [
        statistic_to_power_stderr_over_time[key] for key in statistic_to_rejections
    ]

    return plot_multiple_curves(
        y_lists,
        steps,
        y_err=y_lists_err,
        keys=list(statistic_to_rejections.keys()),
        xlabel="Step",
        ylabel="Power",
        title=title,
        show=True,
        save_path=save_path,
    )
