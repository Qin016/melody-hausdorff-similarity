from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

SUBSET_SIZE = 25
POINTS_CSV = PROCESSED_DIR / f"bodhidharma_subset_{SUBSET_SIZE}_melody_points.csv"
SUMMARY_CSV = PROCESSED_DIR / "sampling_points_ablation_summary.csv"
PREDICTIONS_CSV = PROCESSED_DIR / "sampling_points_ablation_predictions.csv"
ACCURACY_PNG = FIGURES_DIR / "sampling_points_ablation_accuracy.png"
DISTANCE_PNG = FIGURES_DIR / "sampling_points_ablation_distances.png"

SAMPLE_SIZES = [100, 200, 300, 500, 800, 1000]
CURVE_COLUMNS = ["time_norm", "pitch_norm", "velocity_norm"]


def evenly_sample(group: pd.DataFrame, max_points: int) -> pd.DataFrame:
    group = group.sort_values("point_index").reset_index(drop=True)
    if len(group) <= max_points:
        sampled = group.copy()
    else:
        indices = np.linspace(0, len(group) - 1, max_points).round().astype(int)
        sampled = group.iloc[indices].copy()
    sampled["sample_index"] = np.arange(1, len(sampled) + 1)
    return sampled


def directed_hausdorff_kdtree(a: np.ndarray, b: np.ndarray) -> float:
    tree = cKDTree(b)
    distances, _ = tree.query(a, k=1, workers=-1)
    return float(distances.max())


def hausdorff_kdtree(a: np.ndarray, b: np.ndarray) -> float:
    return max(directed_hausdorff_kdtree(a, b), directed_hausdorff_kdtree(b, a))


def build_sampled_curves(points: pd.DataFrame, sample_size: int) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    keys = ["song_id", "genre", "file_name", "relative_path"]
    sampled_groups = []
    for _, group in points.groupby(keys, sort=False):
        sampled_groups.append(evenly_sample(group, sample_size))
    sampled = pd.concat(sampled_groups, ignore_index=True)

    songs = (
        sampled[keys]
        .drop_duplicates()
        .sort_values(["genre", "file_name"])
        .reset_index(drop=True)
    )

    curves: dict[int, np.ndarray] = {}
    for song_id, group in sampled.sort_values(["song_id", "sample_index"]).groupby("song_id"):
        curves[int(song_id)] = group[CURVE_COLUMNS].to_numpy(float)

    return songs, curves


def compute_distance_matrix(songs: pd.DataFrame, curves: dict[int, np.ndarray]) -> np.ndarray:
    song_ids = songs["song_id"].astype(int).to_list()
    n = len(song_ids)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        curve_i = curves[song_ids[i]]
        for j in range(i + 1, n):
            distance = hausdorff_kdtree(curve_i, curves[song_ids[j]])
            matrix[i, j] = distance
            matrix[j, i] = distance
    return matrix


def evaluate_1nn(songs: pd.DataFrame, matrix: np.ndarray, sample_size: int) -> pd.DataFrame:
    rows = []
    for i, song in songs.iterrows():
        distances = matrix[i].copy()
        distances[i] = np.inf
        nearest_idx = int(distances.argmin())
        nearest = songs.iloc[nearest_idx]
        rows.append(
            {
                "sample_size": sample_size,
                "song_id": int(song["song_id"]),
                "genre": song["genre"],
                "file_name": song["file_name"],
                "nearest_song_id": int(nearest["song_id"]),
                "nearest_genre": nearest["genre"],
                "nearest_file_name": nearest["file_name"],
                "distance": float(distances[nearest_idx]),
                "correct": song["genre"] == nearest["genre"],
            }
        )
    return pd.DataFrame(rows)


def summarize_distances(songs: pd.DataFrame, matrix: np.ndarray, sample_size: int, elapsed_seconds: float) -> dict[str, float]:
    genres = songs["genre"].to_numpy()
    upper_i, upper_j = np.triu_indices(len(songs), k=1)
    distances = matrix[upper_i, upper_j]
    same_mask = genres[upper_i] == genres[upper_j]
    same = distances[same_mask]
    different = distances[~same_mask]
    return {
        "sample_size": sample_size,
        "songs": len(songs),
        "pairwise_distances": int(len(distances)),
        "same_genre_mean": float(same.mean()),
        "same_genre_median": float(np.median(same)),
        "different_genre_mean": float(different.mean()),
        "different_genre_median": float(np.median(different)),
        "mean_gap": float(different.mean() - same.mean()),
        "median_gap": float(np.median(different) - np.median(same)),
        "elapsed_seconds": elapsed_seconds,
    }


def plot_results(summary: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(summary["sample_size"], summary["accuracy"], marker="o", linewidth=2)
    plt.xlabel("Max sampled points per song")
    plt.ylabel("1-NN accuracy")
    plt.title("Sampling Points Ablation: 1-NN Accuracy")
    plt.ylim(0, max(0.5, summary["accuracy"].max() + 0.08))
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(ACCURACY_PNG, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(summary["sample_size"], summary["same_genre_mean"], marker="o", label="Same genre")
    plt.plot(summary["sample_size"], summary["different_genre_mean"], marker="o", label="Different genre")
    plt.bar(summary["sample_size"], summary["mean_gap"], alpha=0.22, label="Mean gap")
    plt.xlabel("Max sampled points per song")
    plt.ylabel("Hausdorff distance")
    plt.title("Sampling Points Ablation: Distance Separation")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(DISTANCE_PNG, dpi=200)
    plt.close()


def main() -> None:
    points = pd.read_csv(POINTS_CSV)
    points = points.sort_values(["song_id", "point_index"])

    summary_rows = []
    prediction_frames = []
    for sample_size in SAMPLE_SIZES:
        start = time.perf_counter()
        songs, curves = build_sampled_curves(points, sample_size)
        matrix = compute_distance_matrix(songs, curves)
        elapsed = time.perf_counter() - start

        predictions = evaluate_1nn(songs, matrix, sample_size)
        summary = summarize_distances(songs, matrix, sample_size, elapsed)
        summary["accuracy"] = float(predictions["correct"].mean())
        summary_rows.append(summary)
        prediction_frames.append(predictions)

        print(
            f"sample_size={sample_size}: "
            f"accuracy={summary['accuracy']:.4f}, "
            f"same_mean={summary['same_genre_mean']:.4f}, "
            f"diff_mean={summary['different_genre_mean']:.4f}, "
            f"gap={summary['mean_gap']:.4f}, "
            f"seconds={elapsed:.1f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")
    predictions_df.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8")
    plot_results(summary_df)

    print(f"summary: {SUMMARY_CSV}")
    print(f"predictions: {PREDICTIONS_CSV}")
    print(f"figures: {ACCURACY_PNG}, {DISTANCE_PNG}")


if __name__ == "__main__":
    main()
