import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np


IEEE_SINGLE_COL_IN = 3.5   # IEEE single-column figure width
IEEE_DOUBLE_COL_IN = 7.16  # IEEE double-column figure width
IEEE_PNG_DPI = 600         # IEEE: >=300 dpi (photos), >=600 dpi (line art)

# Slightly smaller (per request) for PNG readability.
FIG_FONT_BASE = 14
FIG_FONT_TICK = 12
FIG_FONT_ANNOT = 10
FIG_FONT_TITLE = 12


def _extract_markdown_table(md_text: str, header_regex: str) -> list[dict[str, str]]:
    """
    Extract a markdown pipe-table that starts right after a header line matching header_regex.
    Returns rows as dicts keyed by column names (raw strings).
    """
    m = re.search(header_regex, md_text, flags=re.MULTILINE)
    if not m:
        raise ValueError(f"Header not found: {header_regex!r}")

    # Find first table line after header
    start = md_text.find("\n", m.end()) + 1
    lines = md_text[start:].splitlines()

    table_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("|"):
            table_lines.append(line.rstrip())
        else:
            if table_lines:
                break

    if len(table_lines) < 3:
        raise ValueError("Table too short / not found after header.")

    # Parse header + separator + rows
    header_cells = [c.strip() for c in table_lines[0].strip("|").split("|")]
    sep = table_lines[1]
    if "-" not in sep:
        raise ValueError("Second line does not look like a separator row.")

    rows: list[dict[str, str]] = []
    for row_line in table_lines[2:]:
        cells = [c.strip() for c in row_line.strip("|").split("|")]
        if len(cells) != len(header_cells):
            # Allow trailing pipes inconsistencies by skipping malformed lines
            continue
        rows.append({header_cells[i]: cells[i] for i in range(len(header_cells))})
    return rows


def _to_float(x: str) -> float:
    # Strip markdown emphasis and unicode minus
    s = x.replace("**", "").replace("−", "-")
    # Take first float-like token
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        raise ValueError(f"Could not parse float from: {x!r}")
    return float(m.group(0))


def _to_int(x: str) -> int:
    s = x.replace("**", "")
    m = re.search(r"\d+", s)
    if not m:
        raise ValueError(f"Could not parse int from: {x!r}")
    return int(m.group(0))


def _clean_label(s: str) -> str:
    return s.replace("**", "").strip()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    summary_path = repo_root / "Ucaps_raw_videos" / "V2.4_TRAINING_SUMMARY.md"
    md_text = summary_path.read_text(encoding="utf-8", errors="ignore")
    # Feature-wise test performance table
    feature_rows = _extract_markdown_table(
        md_text, r"^### Individual Feature Performance \(Test - Ensemble\)\s*$"
    )

    # Parse features (keep same order as the user figure)
    feature_order = [
        "Nostril_muzzle",
        "Lip_jaw_profile",
        "Ears_lateral",
        "Ears_frontal",
        "Cheek_tightening",
        "Tension_above_eyes",
        "Orbital_tightening",
    ]
    feature_labels = {
        "Nostril_muzzle": "Nostril/muzzle",
        "Lip_jaw_profile": "Lip/jaw profile",
        "Ears_lateral": "Ears lateral",
        "Ears_frontal": "Ears frontal",
        "Cheek_tightening": "Cheek tightening",
        "Tension_above_eyes": "Tension above eyes",
        "Orbital_tightening": "Orbital tightening",
    }
    feature_r2_map = {}
    for r in feature_rows:
        feat_raw = _clean_label(r.get("Feature", ""))
        # Feature names in table like "**Orbital_tightening**"
        feat_key = feat_raw
        feature_r2_map[feat_key] = _to_float(r["R²"])

    r2 = np.array([feature_r2_map[k] for k in feature_order], dtype=float)

    # Style (IEEE-friendly)
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.antialiased": True,
            "font.size": FIG_FONT_BASE,
            "axes.titlesize": FIG_FONT_TITLE,
            "axes.labelsize": FIG_FONT_BASE,
            "xtick.labelsize": FIG_FONT_TICK,
            "ytick.labelsize": FIG_FONT_TICK,
        }
    )

    # Single-panel figure (feature-wise R^2 only) to avoid label crowding.
    fig, ax_b = plt.subplots(figsize=(IEEE_DOUBLE_COL_IN, 3.8), dpi=300)
    y = np.arange(len(feature_order))
    feat_names = [feature_labels[k] for k in feature_order]
    green = "#54A24B"  # colorblind-safe green
    # Slightly thinner bars for cleaner labels/spacing.
    # Keep all bars the same height; avoid altering perceived values.
    ax_b.barh(y, r2, height=0.62, color=green, edgecolor="black", linewidth=0.6)
    ax_b.axvline(0.0, color="#444444", linewidth=0.9)
    ax_b.set_yticks(y, feat_names)
    ax_b.set_xlabel(r"$R^2$ (test)")
    ax_b.set_title(r"Feature-wise $R^2$ (test)")
    # Light grid behind everything (avoid fighting with text/labels)
    ax_b.grid(axis="x", color="#D0D0D0", linestyle="-", linewidth=0.7, alpha=0.7, zorder=0)
    ax_b.tick_params(axis="x", labelsize=FIG_FONT_TICK)
    ax_b.tick_params(axis="y", labelsize=FIG_FONT_BASE)

    # Value labels (keep outside bar ends, avoid clipping)
    pos_offset = 0.008  # pull positives inward (avoid right-edge clipping)
    neg_offset_default = 0.010
    # Special-case: keep "Nostril/muzzle" label outside the bar end, but we will
    # expand the left x-limit so it stays inside the axes (no overlap with y-labels).
    neg_offset_nostril = 0.006
    for i in range(len(r2)):
        v = r2[i]
        if v >= 0:
            # Place positive labels just to the right of the bar end.
            x_text = v + pos_offset
            ha = "left"
        else:
            # Place negative labels just to the left of the bar end (outside the bar),
            # but rely on extra left margin to avoid colliding with y tick labels.
            feat_key = feature_order[i]
            neg_offset = neg_offset_nostril if feat_key == "Nostril_muzzle" else neg_offset_default
            x_text = v - neg_offset
            ha = "right"
        ax_b.text(
            x_text,
            i,
            f"{v:.3f}",
            va="center",
            ha=ha,
            fontsize=FIG_FONT_ANNOT,
            clip_on=False,
            zorder=5,
            # Clean readability without "boxed" artifacts
            path_effects=[pe.withStroke(linewidth=3, foreground="white"), pe.Normal()],
        )

    # Make a bit of room for negative labels
    # Expand left side so the most-negative label fits fully inside the axes.
    xmin = min(-0.18, float(np.min(r2)) - 0.08)
    xmax = max(0.34, float(np.max(r2)) + 0.07)
    ax_b.set_xlim(xmin, xmax)

    # Increase left/right margins so long labels and right-edge values don't clip.
    fig.subplots_adjust(left=0.44, right=0.995, top=0.88, bottom=0.22)

    out_dir = Path(__file__).resolve().parent / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "fig8_regression_summary_v2_4.png"
    out_pdf = out_dir / "fig8_regression_summary_v2_4.pdf"
    out_svg = out_dir / "fig8_regression_summary_v2_4.svg"

    # Preferred for IEEE: vector formats (crisp at any zoom).
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.01)

    # High-res raster for previews; keep opaque white to avoid alpha fringing.
    fig.savefig(
        out_png,
        dpi=IEEE_PNG_DPI,
        bbox_inches="tight",
        pad_inches=0.01,
        facecolor="white",
        transparent=False,
    )
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")
    print(f"Wrote: {out_svg}")


if __name__ == "__main__":
    main()


