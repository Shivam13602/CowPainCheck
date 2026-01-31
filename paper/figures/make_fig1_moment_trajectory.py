import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


IEEE_SINGLE_COL_IN = 3.5   # IEEE single-column figure width
IEEE_DOUBLE_COL_IN = 7.16  # IEEE double-column figure width
IEEE_PNG_DPI = 600         # IEEE: >=300 dpi (photos), >=600 dpi (line art)

# Slightly smaller than the previous iteration, still highly readable.
FIG_FONT_BASE = 16
FIG_FONT_TICK = 14
FIG_FONT_LEGEND = 13


def _coerce_numeric(series: pd.Series) -> pd.Series:
    # The UCAPS tables use "." to indicate missing entries.
    return pd.to_numeric(series.replace({".": pd.NA, "": pd.NA}), errors="coerce")


def _mean_ci95_by_moment(df: pd.DataFrame, moment_col: str, moments: list[str], col: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute mean and (approx.) 95% CI of the mean per moment using normal approximation:
        CI = mean ± 1.96 * (std / sqrt(n))
    """
    g = df.groupby(moment_col, dropna=False)[col]
    mean = g.mean().reindex(moments)
    std = g.std(ddof=1).reindex(moments)
    n = g.count().reindex(moments)
    se = std / (n ** 0.5)
    ci = 1.96 * se
    lo = mean - ci
    hi = mean + ci
    return mean, lo, hi


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_csv = os.path.join(repo_root, "Ucaps_raw_videos", "Data_real_time_bovine_Unesp_2019_RMT.csv")
    out_dir = os.path.join(os.path.dirname(__file__), "exports")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(src_csv)

    moment_col = "Time-point"
    moments = ["M0", "M1", "M2", "M3", "M4"]

    # Columns in the UNESP facial-action table
    fa_cols = {
        "Orbital tightening": "1.Orbital.tightening",
        "Tension above eyes": "2.Tension.above.eyes",
        "Cheek tightening": "3.Cheek.(masseter.muscle).tightnening",
        "Ears (frontal)": "4.Ears.Position.Frontal",
        "Ears (lateral)": "5.Ears.Position.Lateral",
        "Lip/Jaw profile": "6.Abnormal.Lip.and.Jaw.Profile",
        "Nostril/Muzzle": "7.Abnormal.Nostril.Muzzle.Shape",
    }
    nrs_col = "NRS (1-10)"

    # Coerce numeric columns
    for col in list(fa_cols.values()) + [nrs_col]:
        df[col] = _coerce_numeric(df[col])

    # Mean + CI per moment (over evaluator-level records; missing entries excluded per column).
    means: dict[str, pd.Series] = {}
    lows: dict[str, pd.Series] = {}
    highs: dict[str, pd.Series] = {}
    for label, col in fa_cols.items():
        m, lo, hi = _mean_ci95_by_moment(df, moment_col, moments, col)
        means[label], lows[label], highs[label] = m, lo, hi

    nrs_mean, nrs_lo, nrs_hi = _mean_ci95_by_moment(df, moment_col, moments, nrs_col)
    # Scale NRS (1–10) to 0–2: 1 -> 0 and 10 -> 2.
    nrs_scaled = (nrs_mean - 1.0) * (2.0 / 9.0)
    nrs_scaled_lo = (nrs_lo - 1.0) * (2.0 / 9.0)
    nrs_scaled_hi = (nrs_hi - 1.0) * (2.0 / 9.0)

    # IEEE two-column friendly styling (legible at final column width).
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
            "axes.labelsize": FIG_FONT_BASE,
            "axes.titlesize": FIG_FONT_BASE,
            "legend.fontsize": FIG_FONT_LEGEND,
            "xtick.labelsize": FIG_FONT_TICK,
            "ytick.labelsize": FIG_FONT_TICK,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # Many series (7 FAUs + NRS): render at IEEE double-column width.
    fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_COL_IN, 4.6), dpi=300, constrained_layout=True)

    x = list(range(len(moments)))
    # Keep tick labels short for two-column layout; details belong in the caption.
    xtick_labels = moments

    # Light-gray acute pain window band + a dashed marker at M2 (matching the reference style).
    ax.axvspan(1.75, 2.25, color="#f3b2b2", alpha=0.30, zorder=0)
    ax.axvline(2.0, color="#cc3b3b", linestyle="--", linewidth=1.0, alpha=0.9, zorder=1)

    # Subtle grid like journal plots (white background).
    ax.grid(True, axis="y", color="0.88", linewidth=0.7)
    ax.grid(False, axis="x")

    # Use a clean palette; uncertainty bands add interpretability.
    colors = [
        "#1f77b4",  # Orbital tightening
        "#ff7f0e",  # Tension above eyes
        "#2ca02c",  # Cheek tightening
        "#d62728",  # Ears frontal
        "#17becf",  # Ears lateral
        "#9467bd",  # Lip/Jaw profile
        "#7f7f7f",  # Nostril/Muzzle
    ]

    for (i, (label, series)) in enumerate(means.items()):
        c = colors[i % len(colors)]
        ax.plot(
            x,
            series.values,
            label=label,
            color=c,
            linewidth=2.2,
            marker="o",
            markersize=7,
            zorder=3,
        )
        ax.fill_between(
            x,
            lows[label].values,
            highs[label].values,
            color=c,
            alpha=0.12,
            linewidth=0.0,
            zorder=2,
        )

    # NRS scaled to 0–2 range with diamond markers (as in the reference figure).
    ax.plot(
        x,
        nrs_scaled.values,
        label="NRS (scaled)",
        color="#111111",
        linewidth=3.0,
        marker="D",
        markersize=8,
        zorder=4,
    )
    ax.fill_between(
        x,
        nrs_scaled_lo.values,
        nrs_scaled_hi.values,
        color="#111111",
        alpha=0.10,
        linewidth=0.0,
        zorder=1,
    )

    ax.set_xticks(x, xtick_labels)
    ax.set_ylabel("Pain cue intensity (0–2 scale)")
    ax.set_xlabel("Time point")
    ax.set_ylim(0, 2.25)
    ax.set_xlim(-0.2, len(moments) - 0.8)

    # No in-figure title (IEEE captions carry the description).
    # Compact legend for two-column layout.
    ax.legend(loc="upper left", ncol=2, frameon=False, columnspacing=0.9, handlelength=1.6)

    out_png = os.path.join(out_dir, "fig1_moment_trajectory_ieee.png")
    out_pdf = os.path.join(out_dir, "fig1_moment_trajectory_ieee.pdf")

    # High-resolution raster + vector for publication workflows.
    fig.savefig(out_png, dpi=IEEE_PNG_DPI, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.01)

    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()


