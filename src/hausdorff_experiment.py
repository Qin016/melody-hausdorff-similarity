from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

SUBSET_SIZE = 25
MAX_POINTS_PER_SONG = 300
POINTS_CSV = PROCESSED_DIR / f"bodhidharma_subset_{SUBSET_SIZE}_melody_points_sampled_{MAX_POINTS_PER_SONG}.csv"
DISTANCE_CSV = PROCESSED_DIR / f"hausdorff_subset_{SUBSET_SIZE}_distance_matrix.csv"
PAIRWISE_CSV = PROCESSED_DIR / f"hausdorff_subset_{SUBSET_SIZE}_pairwise_distances.csv"
GENRE_SUMMARY_CSV = PROCESSED_DIR / f"hausdorff_subset_{SUBSET_SIZE}_genre_summary.csv"
KNN_CSV = PROCESSED_DIR / f"hausdorff_subset_{SUBSET_SIZE}_1nn_predictions.csv"
HEATMAP_PNG = FIGURES_DIR / "hausdorff_distance_heatmap.png"
BOXPLOT_PNG = FIGURES_DIR / "hausdorff_same_vs_diff_boxplot.png"
DENDROGRAM_PNG = FIGURES_DIR / "hausdorff_hierarchical_clustering.png"


def directed_hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    min_distances = []
    chunk_size = 64
    for start in range(0, a.shape[0], chunk_size):
        chunk = a[start : start + chunk_size]
        distances = np.linalg.norm(chunk[:, None, :] - b[None, :, :], axis=2)
        min_distances.append(distances.min(axis=1))
    return float(np.concatenate(min_distances).max())


def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    return max(directed_hausdorff(a, b), directed_hausdorff(b, a))


def load_curves() -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    points = pd.read_csv(POINTS_CSV)
    points = points.sort_values(["song_id", "sample_index"])
    songs = (
        points[["song_id", "genre", "file_name", "relative_path"]]
        .drop_duplicates()
        .sort_values(["genre", "file_name"])
        .reset_index(drop=True)
    )

    curves: dict[int, np.ndarray] = {}
    for song_id, group in points.groupby("song_id"):
        curves[int(song_id)] = group[["time_norm", "pitch_norm", "velocity_norm"]].to_numpy(float)

    return songs, curves


def compute_distances(songs: pd.DataFrame, curves: dict[int, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(songs)
    matrix = np.zeros((n, n), dtype=float)
    pair_rows: list[dict[str, object]] = []

    for i in range(n):
        song_i = songs.iloc[i]
        curve_i = curves[int(song_i["song_id"])]
        for j in range(i + 1, n):
            song_j = songs.iloc[j]
            curve_j = curves[int(song_j["song_id"])]
            distance = hausdorff(curve_i, curve_j)
            matrix[i, j] = distance
            matrix[j, i] = distance
            pair_rows.append(
                {
                    "song_id_a": int(song_i["song_id"]),
                    "genre_a": song_i["genre"],
                    "file_name_a": song_i["file_name"],
                    "song_id_b": int(song_j["song_id"]),
                    "genre_b": song_j["genre"],
                    "file_name_b": song_j["file_name"],
                    "same_genre": song_i["genre"] == song_j["genre"],
                    "hausdorff_distance": distance,
                }
            )

    labels = [f"{row.genre} | {row.file_name}" for row in songs.itertuples(index=False)]
    distance_df = pd.DataFrame(matrix, index=labels, columns=labels)
    pairwise_df = pd.DataFrame(pair_rows)
    return distance_df, pairwise_df


def summarize_pairs(pairwise: pd.DataFrame) -> pd.DataFrame:
    same_diff = (
        pairwise.assign(group=lambda df: np.where(df["same_genre"], "same_genre", "different_genre"))
        .groupby("group")["hausdorff_distance"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

    by_pair = (
        pairwise.assign(
            genre_pair=lambda df: np.where(
                df["genre_a"] <= df["genre_b"],
                df["genre_a"] + " / " + df["genre_b"],
                df["genre_b"] + " / " + df["genre_a"],
            )
        )
        .groupby("genre_pair")["hausdorff_distance"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )

    return pd.concat(
        [
            pd.DataFrame([{"section": "same_vs_different"}]),
            same_diff.assign(section="same_vs_different"),
            pd.DataFrame([{"section": "genre_pairs"}]),
            by_pair.assign(section="genre_pairs"),
        ],
        ignore_index=True,
        sort=False,
    )


def one_nearest_neighbor(songs: pd.DataFrame, distance_df: pd.DataFrame) -> pd.DataFrame:
    distances = distance_df.to_numpy()
    rows = []
    for i, song in songs.iterrows():
        row = distances[i].copy()
        row[i] = np.inf
        nearest_idx = int(row.argmin())
        nearest = songs.iloc[nearest_idx]
        rows.append(
            {
                "song_id": int(song["song_id"]),
                "genre": song["genre"],
                "file_name": song["file_name"],
                "nearest_song_id": int(nearest["song_id"]),
                "nearest_genre": nearest["genre"],
                "nearest_file_name": nearest["file_name"],
                "distance": float(row[nearest_idx]),
                "correct": song["genre"] == nearest["genre"],
            }
        )
    return pd.DataFrame(rows)


def plot_heatmap(distance_df: pd.DataFrame, songs: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    genres = songs["genre"].tolist()
    short_labels = [f"{genre[:3]}-{idx + 1:02d}" for idx, genre in enumerate(genres)]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        distance_df,
        cmap="viridis",
        xticklabels=short_labels,
        yticklabels=short_labels,
        square=True,
        cbar_kws={"label": "Hausdorff distance"},
    )
    plt.title("Hausdorff Distance Matrix for Sampled Melody Curves")
    plt.xlabel("Songs")
    plt.ylabel("Songs")
    plt.tight_layout()
    plt.savefig(HEATMAP_PNG, dpi=200)
    plt.close()


def plot_boxplot(pairwise: pd.DataFrame) -> None:
    plot_df = pairwise.copy()
    plot_df["comparison"] = np.where(plot_df["same_genre"], "Same genre", "Different genre")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=plot_df, x="comparison", y="hausdorff_distance", hue="comparison", palette="Set2", legend=False)
    sns.stripplot(data=plot_df, x="comparison", y="hausdorff_distance", color="black", alpha=0.25, size=2)
    plt.title("Hausdorff Distance: Same Genre vs Different Genre")
    plt.xlabel("")
    plt.ylabel("Hausdorff distance")
    plt.tight_layout()
    plt.savefig(BOXPLOT_PNG, dpi=200)
    plt.close()


def plot_dendrogram(distance_df: pd.DataFrame, songs: pd.DataFrame) -> None:
    labels = [f"{row.genre[:4]}-{idx + 1:02d}" for idx, row in songs.iterrows()]
    condensed = squareform(distance_df.to_numpy(), checks=False)
    linked = linkage(condensed, method="average")

    plt.figure(figsize=(16, 7))
    dendrogram(linked, labels=labels, leaf_rotation=90, leaf_font_size=7)
    plt.title("Hierarchical Clustering Based on Hausdorff Distance")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(DENDROGRAM_PNG, dpi=200)
    plt.close()


def main() -> None:
    songs, curves = load_curves()
    distance_df, pairwise_df = compute_distances(songs, curves)
    summary_df = summarize_pairs(pairwise_df)
    knn_df = one_nearest_neighbor(songs, distance_df)

    distance_df.to_csv(DISTANCE_CSV, encoding="utf-8")
    pairwise_df.to_csv(PAIRWISE_CSV, index=False, encoding="utf-8")
    summary_df.to_csv(GENRE_SUMMARY_CSV, index=False, encoding="utf-8")
    knn_df.to_csv(KNN_CSV, index=False, encoding="utf-8")

    plot_heatmap(distance_df, songs)
    plot_boxplot(pairwise_df)
    plot_dendrogram(distance_df, songs)

    same_mean = pairwise_df.loc[pairwise_df["same_genre"], "hausdorff_distance"].mean()
    diff_mean = pairwise_df.loc[~pairwise_df["same_genre"], "hausdorff_distance"].mean()
    accuracy = knn_df["correct"].mean()

    print(f"songs: {len(songs)}")
    print(f"pairwise distances: {len(pairwise_df)}")
    print(f"same-genre mean distance: {same_mean:.4f}")
    print(f"different-genre mean distance: {diff_mean:.4f}")
    print(f"1-NN genre accuracy: {accuracy:.4f}")
    print(f"distance matrix: {DISTANCE_CSV}")
    print(f"pairwise distances: {PAIRWISE_CSV}")
    print(f"summary: {GENRE_SUMMARY_CSV}")
    print(f"1-NN predictions: {KNN_CSV}")
    print(f"figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
