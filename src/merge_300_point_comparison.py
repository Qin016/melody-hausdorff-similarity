from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

ABLATION_CSV = PROCESSED_DIR / "sampling_points_ablation_summary.csv"
DISTANCE_300_CSV = PROCESSED_DIR / "distance_method_comparison_summary_300.csv"
ML_CSV = PROCESSED_DIR / "ml_classifier_comparison_summary.csv"
OUTPUT_CSV = PROCESSED_DIR / "method_comparison_summary_300_combined.csv"
OUTPUT_PNG = FIGURES_DIR / "method_comparison_accuracy_300_combined.png"


def main() -> None:
    ablation = pd.read_csv(ABLATION_CSV)
    hausdorff_300 = ablation.loc[ablation["sample_size"] == 300].iloc[0]
    distance = pd.read_csv(DISTANCE_300_CSV)
    ml = pd.read_csv(ML_CSV)

    distance_rows = pd.DataFrame(
        [
            {
                "family": "Distance 1-NN",
                "method": "Hausdorff",
                "accuracy": float(hausdorff_300["accuracy"]),
                "same_genre_mean": float(hausdorff_300["same_genre_mean"]),
                "different_genre_mean": float(hausdorff_300["different_genre_mean"]),
                "mean_gap": float(hausdorff_300["mean_gap"]),
            }
        ]
    )
    distance_rows = pd.concat(
        [
            distance_rows,
            distance.assign(family="Distance 1-NN").rename(columns={"accuracy": "accuracy"})[
                ["family", "method", "accuracy", "same_genre_mean", "different_genre_mean", "mean_gap"]
            ],
        ],
        ignore_index=True,
    )

    ml_rows = ml.assign(
        family="ML classifier",
        accuracy=ml["accuracy_mean"],
        same_genre_mean=pd.NA,
        different_genre_mean=pd.NA,
        mean_gap=pd.NA,
    )[["family", "method", "accuracy", "same_genre_mean", "different_genre_mean", "mean_gap"]]

    output = pd.concat([distance_rows, ml_rows], ignore_index=True)
    output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    plot_df = output.sort_values("accuracy", ascending=True)
    colors = plot_df["family"].map({"Distance 1-NN": "#ff9f1c", "ML classifier": "#2ec4b6"})
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["method"], plot_df["accuracy"], color=colors)
    plt.xlabel("Accuracy")
    plt.title("300-Point Distance Methods vs Machine Learning Classifiers")
    plt.xlim(0, max(0.65, float(plot_df["accuracy"].max()) + 0.08))
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200)
    plt.close()

    print(output.to_string(index=False))
    print(f"summary: {OUTPUT_CSV}")
    print(f"figure: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
