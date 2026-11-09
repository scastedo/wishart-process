"""Plot linear/quadratic fractions after cumulatively averaging saved windows.

This script uses the existing per-window Wishart outputs. It does not treat the
cumulative averages as new Wishart fits to longer raw-data averaging windows.
"""

from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls


RESULT_DIR = Path("wishart_jun3_dynamic_equal40_sated")
OUTPUT_FIGURE = Path("cumulative_window_linear_quadratic.png")
OUTPUT_SUMMARY = Path("cumulative_window_linear_quadratic_summary.csv")
OUTPUT_FITS = Path("cumulative_window_linear_quadratic_animal_fits.csv")

TARGET_GAP = 30.0
FRAME_RATE = 20.0
MODE_START = 1  # Exclude the largest covariance eigenmode.


def load_mode_data():
    rows = []

    for path in sorted(RESULT_DIR.glob("*_overlaps_eigs.pkl")):
        with path.open("rb") as handle:
            saved = pickle.load(handle)

        gaps = np.atleast_1d(np.asarray(saved["small_angle"], dtype=float))
        gap_indices = np.flatnonzero(np.isclose(gaps, TARGET_GAP))
        if len(gap_indices) != 1:
            continue

        overlaps = np.asarray(saved["overlaps_small"], dtype=float)
        eigenvalues = np.asarray(saved["eigs_per_condition"], dtype=float)

        if overlaps.ndim == 4:
            overlaps = overlaps[None, ...]
        if eigenvalues.ndim == 3:
            eigenvalues = eigenvalues[..., None]

        # Average angle origins, spatial frequencies, and the internal axis.
        overlap_by_mode = np.nanmean(
            overlaps[gap_indices[0]],
            axis=(0, 1, 3),
        )
        eigenvalue_by_mode = np.nanmean(
            eigenvalues,
            axis=(0, 1, 3),
        )

        group = "FR" if path.name.startswith("FR_") else "CTR"
        start = int(saved["start"])
        stop = int(saved["stop"])

        for mode in range(MODE_START, len(overlap_by_mode)):
            rows.append(
                {
                    "group": group,
                    "animal": int(saved["animal"]),
                    "repeat": int(saved["repeat_idx"]),
                    "start": start,
                    "stop": stop,
                    "mode": mode,
                    "overlap": float(overlap_by_mode[mode]),
                    "eigenvalue": float(eigenvalue_by_mode[mode]),
                }
            )

    if not rows:
        raise RuntimeError(f"No compatible result files found in {RESULT_DIR}")

    return pd.DataFrame(rows)


def fit_cumulative_fractions(data):
    # Ten neuron-subsampling runs are technical repeats, so average them first.
    animal_windows = (
        data.groupby(
            ["group", "animal", "start", "stop", "mode"],
            as_index=False,
        )[["overlap", "eigenvalue"]]
        .mean()
    )

    windows = (
        animal_windows[["start", "stop"]]
        .drop_duplicates()
        .sort_values(["start", "stop"])
        .reset_index(drop=True)
    )
    windows["window_number"] = np.arange(1, len(windows) + 1)
    first_start = int(windows["start"].iloc[0])
    windows["cumulative_span_s"] = (
        windows["stop"] - first_start
    ) / FRAME_RATE

    animal_windows = animal_windows.merge(
        windows[["start", "stop", "window_number"]],
        on=["start", "stop"],
        how="left",
        validate="many_to_one",
    )

    fit_rows = []

    for (group, animal), animal_data in animal_windows.groupby(
        ["group", "animal"]
    ):
        available = np.sort(animal_data["window_number"].unique())
        expected = windows["window_number"].to_numpy()
        if not np.array_equal(available, expected):
            raise RuntimeError(
                f"{group} animal {animal} does not contain every window"
            )

        for number in expected:
            cumulative = (
                animal_data[animal_data["window_number"] <= number]
                .groupby("mode", as_index=False)[["overlap", "eigenvalue"]]
                .mean()
            )

            valid = (
                np.isfinite(cumulative["overlap"])
                & np.isfinite(cumulative["eigenvalue"])
                & (cumulative["overlap"] > 0)
                & (cumulative["eigenvalue"] > 0)
            )
            cumulative = cumulative.loc[valid]

            overlap = cumulative["overlap"].to_numpy()
            eigenvalue = cumulative["eigenvalue"].to_numpy()
            if len(overlap) < 3:
                continue

            design = np.column_stack([overlap, np.sqrt(overlap)])
            alpha, beta = nnls(design, eigenvalue)[0]

            quadratic = alpha * overlap
            linear = beta * np.sqrt(overlap)
            fitted = quadratic + linear
            fitted_total = fitted.sum()
            if not np.isfinite(fitted_total) or fitted_total <= 0:
                continue

            prediction = design @ np.array([alpha, beta])
            residual_ss = np.sum((eigenvalue - prediction) ** 2)
            total_ss = np.sum((eigenvalue - eigenvalue.mean()) ** 2)
            r_squared = 1 - residual_ss / total_ss if total_ss > 0 else np.nan

            span = float(
                windows.loc[
                    windows["window_number"] == number,
                    "cumulative_span_s",
                ].iloc[0]
            )

            fit_rows.append(
                {
                    "group": group,
                    "animal": animal,
                    "windows_averaged": int(number),
                    "cumulative_span_s": span,
                    "alpha": alpha,
                    "beta": beta,
                    "linear_fraction": linear.sum() / fitted_total,
                    "quadratic_fraction": quadratic.sum() / fitted_total,
                    "r_squared": r_squared,
                }
            )

    fits = pd.DataFrame(fit_rows)
    if fits.empty:
        raise RuntimeError("No cumulative fits could be calculated")

    summary = (
        fits.groupby(
            ["group", "windows_averaged", "cumulative_span_s"],
            as_index=False,
        )
        .agg(
            linear_mean=("linear_fraction", "mean"),
            linear_sem=("linear_fraction", "sem"),
            quadratic_mean=("quadratic_fraction", "mean"),
            quadratic_sem=("quadratic_fraction", "sem"),
            r_squared_mean=("r_squared", "mean"),
            n_animals=("animal", "nunique"),
        )
    )

    return fits, summary


def plot_summary(summary):
    colors = {"FR": "tab:red", "CTR": "tab:blue"}
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.4),
        dpi=200,
        sharex=True,
        sharey=True,
    )

    for axis, term, title in zip(
        axes,
        ("linear", "quadratic"),
        ("Linear contribution", "Quadratic contribution"),
    ):
        for group, color in colors.items():
            group_summary = summary[
                summary["group"] == group
            ].sort_values("windows_averaged")

            x = group_summary["windows_averaged"].to_numpy()
            mean = group_summary[f"{term}_mean"].to_numpy()
            sem = group_summary[f"{term}_sem"].to_numpy()

            axis.plot(
                x,
                mean,
                color=color,
                marker="o",
                linewidth=2,
                markersize=4,
                label=group,
            )
            axis.fill_between(
                x,
                mean - sem,
                mean + sem,
                color=color,
                alpha=0.16,
                linewidth=0,
            )

        axis.axhline(0.5, color="0.4", linestyle=":", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Number of consecutive fitted windows averaged")
        axis.set_xticks(np.sort(summary["windows_averaged"].unique()))
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)

    axes[0].set_ylabel("Fraction of fitted covariance-eigenvalue term")
    fig.suptitle(
        rf"Cumulative averaging of 0.25-s Wishart summaries; "
        rf"$\Delta\theta={TARGET_GAP:g}^\circ$, largest mode excluded"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, bbox_inches="tight")
    plt.close(fig)


def main():
    data = load_mode_data()
    fits, summary = fit_cumulative_fractions(data)
    fits.to_csv(OUTPUT_FITS, index=False)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    plot_summary(summary)

    endpoints = summary[
        summary["windows_averaged"].isin(
            [summary["windows_averaged"].min(), summary["windows_averaged"].max()]
        )
    ]
    print(endpoints.to_string(index=False))
    print(f"Saved figure: {OUTPUT_FIGURE.resolve()}")
    print(f"Saved summary: {OUTPUT_SUMMARY.resolve()}")
    print(f"Saved animal fits: {OUTPUT_FITS.resolve()}")


if __name__ == "__main__":
    main()
