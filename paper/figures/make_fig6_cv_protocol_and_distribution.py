import os
import re
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class SplitSpec:
    test_animals: tuple[int, ...]
    folds_to_val_animals: dict[int, tuple[int, int]]


MOMENTS = ("M0", "M1", "M2", "M3", "M4")


def _count_sequences_in_video_folder(video_folder: str) -> int:
    """
    Count number of sequence_* subfolders under a single video folder.
    """
    try:
        items = os.listdir(video_folder)
    except FileNotFoundError:
        return 0
    return sum(1 for name in items if os.path.isdir(os.path.join(video_folder, name)) and name.lower().startswith("sequence_"))


def _parse_animal_and_moment_from_folder_name(folder_name: str) -> tuple[int | None, str | None]:
    """
    Parse 'Animal <id> M<0-4>' from folder name.
    Examples:
      'Animal 9 M2 Facial 3' -> (9, 'M2')
      'Animal 21 M2 Facial (low video resolution)' -> (21, 'M2')
    """
    m = re.search(r"^Animal\s+(\d+)\s+(M[0-4])\b", folder_name)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2)


def load_fold_splits_from_v24_summary(repo_root: str) -> SplitSpec:
    """
    Extract fold -> [a,b] validation animals from V2.4_TRAINING_SUMMARY.md.
    """
    path = os.path.join(repo_root, "Ucaps_raw_videos", "V2.4_TRAINING_SUMMARY.md")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Rows like: | **0** | [22, 9] | **1.2708** | ...
    rx = re.compile(r"^\|\s+\*\*(\d+)\*\*\s+\|\s+\[(\d+),\s*(\d+)\]\s+\|", re.MULTILINE)
    folds_to_val: dict[int, tuple[int, int]] = {}
    for fold, a, b in rx.findall(text):
        folds_to_val[int(fold)] = (int(a), int(b))

    if len(folds_to_val) != 9:
        raise RuntimeError(f"Expected 9 folds in v2.4 summary; found {len(folds_to_val)} folds.")

    # Paper protocol: held-out test animals
    return SplitSpec(test_animals=(14, 17), folds_to_val_animals=folds_to_val)


def count_sequences_by_animal_and_moment(repo_root: str) -> dict[tuple[int, str], int]:
    """
    Count sequences (sequence_* directories) for each (animal, moment) from sequence/ folder.
    """
    seq_root = os.path.join(repo_root, "sequence")
    counts: dict[tuple[int, str], int] = {}

    for name in os.listdir(seq_root):
        full = os.path.join(seq_root, name)
        if not os.path.isdir(full):
            continue
        animal, moment = _parse_animal_and_moment_from_folder_name(name)
        if animal is None or moment is None:
            continue
        counts[(animal, moment)] = counts.get((animal, moment), 0) + _count_sequences_in_video_folder(full)

    return counts


def _sum_counts_for_animals(counts: dict[tuple[int, str], int], animals: set[int]) -> dict[str, int]:
    out = {m: 0 for m in MOMENTS}
    for (animal, moment), n in counts.items():
        if animal in animals and moment in out:
            out[moment] += int(n)
    return out


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(os.path.dirname(__file__), "exports")
    os.makedirs(out_dir, exist_ok=True)

    split = load_fold_splits_from_v24_summary(repo_root)
    counts = count_sequences_by_animal_and_moment(repo_root)

    animals_in_sequences = sorted({a for (a, _m) in counts.keys()})
    test_animals = set(split.test_animals)

    # CV pool = animals in fold definitions, excluding held-out test animals
    cv_animals = sorted({a for pair in split.folds_to_val_animals.values() for a in pair})
    cv_animals_set = set(cv_animals)

    # Sanity checks
    missing_in_seq = sorted(list(cv_animals_set.union(test_animals) - set(animals_in_sequences)))
    if missing_in_seq:
        raise RuntimeError(f"Some animals referenced by CV/test splits are missing in sequence/: {missing_in_seq}")

    # ---------- Build fold x animal matrix for protocol schematic ----------
    # Animals displayed: CV pool animals (18) + held-out test animals (2)
    display_animals = sorted(cv_animals_set.union(test_animals))
    animal_to_col = {a: i for i, a in enumerate(display_animals)}

    # Values: 0=train, 1=val, 2=test
    mat = np.zeros((9, len(display_animals)), dtype=int)
    for fold in range(9):
        val_pair = split.folds_to_val_animals[fold]
        for a in val_pair:
            mat[fold, animal_to_col[a]] = 1
        for a in test_animals:
            mat[fold, animal_to_col[a]] = 2

    # ---------- Compute distributions ----------
    # Per-fold distributions (train/val), then average across folds.
    train_by_fold = []
    val_by_fold = []
    for fold in range(9):
        val_pair = set(split.folds_to_val_animals[fold])
        train_set = cv_animals_set - val_pair
        train_by_fold.append(_sum_counts_for_animals(counts, train_set))
        val_by_fold.append(_sum_counts_for_animals(counts, val_pair))

    def _avg_dict(dicts: list[dict[str, int]]) -> dict[str, float]:
        out = {}
        for m in MOMENTS:
            out[m] = float(np.mean([d[m] for d in dicts]))
        return out

    train_avg = _avg_dict(train_by_fold)
    val_avg = _avg_dict(val_by_fold)
    test_dist = _sum_counts_for_animals(counts, test_animals)

    # ---------- Plot ----------
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(10.8, 5.4), dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.22)

    # (a) Protocol schematic
    ax0 = fig.add_subplot(gs[0, 0])
    cmap = mpl.colors.ListedColormap(["#d9d9d9", "#4c78a8", "#f58518"])  # train, val, test
    norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    im = ax0.imshow(mat, aspect="auto", cmap=cmap, norm=norm)

    ax0.set_title("Animal-wise Cross-Validation Protocol (9 folds)")
    ax0.set_xlabel("Animal ID")
    ax0.set_ylabel("Fold")
    ax0.set_yticks(range(9), [f"Fold {i}" for i in range(9)])
    ax0.set_xticks(range(len(display_animals)), [str(a) for a in display_animals], rotation=0)

    # Light grid lines
    ax0.set_xticks(np.arange(-0.5, len(display_animals), 1), minor=True)
    ax0.set_yticks(np.arange(-0.5, 9, 1), minor=True)
    ax0.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax0.tick_params(which="minor", bottom=False, left=False)

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#d9d9d9", edgecolor="none", label="Train animals (per fold)"),
        Patch(facecolor="#4c78a8", edgecolor="none", label="Validation animals (2 per fold)"),
        Patch(facecolor="#f58518", edgecolor="none", label=f"Held-out test animals ({', '.join(map(str, split.test_animals))})"),
    ]
    ax0.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False)

    # (b) Moment distribution (avg train/val across folds + fixed test)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title("Sequence Distribution by Moment")

    labels = ["Train (avg)", "Val (avg)", "Test"]
    x = np.arange(len(labels))
    width = 0.6

    moment_colors = {
        "M0": "#4c78a8",
        "M1": "#72b7b2",
        "M2": "#f58518",
        "M3": "#54a24b",
        "M4": "#b279a2",
    }

    bottoms = np.zeros(len(labels))
    for m in MOMENTS:
        vals = np.array([train_avg[m], val_avg[m], float(test_dist[m])], dtype=float)
        ax1.bar(x, vals, width, bottom=bottoms, color=moment_colors[m], label=m)
        bottoms += vals

    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Number of sequences (10 s clips)")
    ax1.grid(True, axis="y", color="0.9", linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(title="Moment", frameon=False, loc="upper right")

    # No figure-level title; IEEE-style captions belong in the manuscript, not embedded in the plot.

    out_png = os.path.join(out_dir, "fig6_cv_protocol_and_distribution.png")
    out_pdf = os.path.join(out_dir, "fig6_cv_protocol_and_distribution.pdf")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()


