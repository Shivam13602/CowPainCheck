import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


IEEE_SINGLE_COL_IN = 3.5   # IEEE single-column figure width
IEEE_DOUBLE_COL_IN = 7.16  # IEEE double-column figure width
IEEE_PNG_DPI = 600         # IEEE: >=300 dpi (photos), >=600 dpi (line art)

# Slightly smaller than the previous iteration, still highly readable.
FIG_FONT_BASE = 16
FIG_FONT_TICK = 14
FIG_FONT_ANNOT = 12
FIG_FONT_TITLE = 14
FIG_FONT_CBAR = 13
FIG_FONT_XTICK = 13


def _plot_cm(ax, cm, title, class_labels):
    cm = np.asarray(cm, dtype=int)
    # Normalize by true class (row) for readability, but keep counts in annotations.
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_title(title, fontsize=FIG_FONT_TITLE)
    # Keep axis labels compact to avoid overlap at large fonts.
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_labels)), class_labels, rotation=0, ha="center")
    ax.set_yticks(range(len(class_labels)), class_labels)
    ax.set_aspect("equal")
    ax.tick_params(axis="x", labelsize=FIG_FONT_XTICK, pad=8)
    ax.tick_params(axis="y", labelsize=FIG_FONT_TICK)

    # Annotate with counts + normalized value.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=FIG_FONT_ANNOT,
                color="white" if cm_norm[i, j] > 0.55 else "black",
            )

    # Light grid
    ax.set_xticks(np.arange(-0.5, len(class_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def main() -> None:
    # Confusion matrices reconstructed from the held-out test classification report in V2.5_TRAINING_SUMMARY.md
    # and the known test supports (N=35): NoPain=14, Acute=9, Residual=12.
    #
    # Task 1 (binary): negatives=14, positives=21 with Recall=1.00 and Precision=0.875
    # => TN=11, FP=3, FN=0, TP=21
    cm_task1 = np.array([[11, 3], [0, 21]], dtype=int)
    labels_task1 = ["No\npain", "Pain"]

    # Task 2 (3-class) report: precision=[1.00,0.41,0.67], recall=[0.71,1.00,0.17], support=[14,9,12]
    # Unique integer CM consistent with these:
    # True NoPain: [10,3,1]
    # True Acute:  [0,9,0]
    # True Resid:  [0,10,2]
    cm_task2 = np.array([[10, 3, 1], [0, 9, 0], [0, 10, 2]], dtype=int)
    # Keep labels compact to remain legible at IEEE column widths.
    # Moment mapping is described in the manuscript caption; keep axis labels short.
    labels_task2 = ["No\npain", "Acute", "Residual"]

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": FIG_FONT_BASE,
            "axes.titlesize": FIG_FONT_TITLE,
            "axes.labelsize": FIG_FONT_BASE,
            "xtick.labelsize": FIG_FONT_TICK,
            "ytick.labelsize": FIG_FONT_TICK,
        }
    )

    # Two panels + colorbar: render at IEEE double-column width.
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(IEEE_DOUBLE_COL_IN, 5.2),
        dpi=300,
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.25]},
    )

    im1 = _plot_cm(ax1, cm_task1, "(a) Binary pain vs. no pain", labels_task1)
    im2 = _plot_cm(ax2, cm_task2, "(b) 3-class intensity-moment", labels_task2)

    # Shared colorbar (normalized scale)
    cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.03, pad=0.02)
    cbar.set_label("Row-normalized", fontsize=FIG_FONT_CBAR)
    cbar.ax.tick_params(labelsize=FIG_FONT_TICK)

    out_dir = os.path.join(os.path.dirname(__file__), "exports")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "fig7_confusion_matrices_v2_5.png")
    out_pdf = os.path.join(out_dir, "fig7_confusion_matrices_v2_5.pdf")
    fig.savefig(out_png, dpi=IEEE_PNG_DPI, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()


