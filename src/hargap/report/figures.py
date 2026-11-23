"""Publication-quality figure generation for Chapters 5 and 6 using matplotlib."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_matplotlib_style():
    """Sets standard academic styling for figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "figure.dpi": 300,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def plot_gap_bars(table_df: pd.DataFrame, output_path: str | Path):
    """
    Renders bar chart comparing P1 macro-F1 vs P2 macro-F1 and generalisation gap.
    """
    set_matplotlib_style()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{r['Dataset']}\n{r['Architecture']}" for _, r in table_df.iterrows()]
    x = np.arange(len(labels))
    width = 0.35

    p1_vals = table_df["Mean P1 F1"].values * 100.0
    p2_vals = table_df["Mean P2 F1"].values * 100.0

    rects1 = ax.bar(x - width/2, p1_vals, width, label="P1 (Record-wise)", color="#3470a3")
    rects2 = ax.bar(x + width/2, p2_vals, width, label="P2 (LOSO)", color="#e07a5f")

    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Cross-Subject Generalisation Gap Across Architectures and Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_subject_boxplots(scores_dict: Dict[str, List[float]], output_path: str | Path):
    """
    Generates per-subject score distribution boxplots for P1 vs P2.
    """
    set_matplotlib_style()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    data = list(scores_dict.values())
    labels = list(scores_dict.keys())

    ax.boxplot(data, patch_artist=True, boxprops=dict(facecolor="#81b29a", color="#3d405b"))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Per-Subject Macro-F1")
    ax.set_title("Subject Score Dispersion: P1 Record-wise vs P2 LOSO")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_heatmap(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str | Path,
    title: str = "LOSO Confusion Matrix"
):
    """
    Plots normalized confusion matrix heatmap.
    """
    set_matplotlib_style()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-8)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Activity",
        xlabel="Predicted Activity",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f"
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm_norm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)
