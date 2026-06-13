from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback for environments without numba
    njit = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

SUBSET_SIZE = 25
MAX_POINTS_PER_SONG = 300
DEFAULT_SEQUENCE_POINTS = 80

POINTS_CSV = PROCESSED_DIR / f"bodhidharma_subset_{SUBSET_SIZE}_melody_points_sampled_{MAX_POINTS_PER_SONG}.csv"
NOTES_CSV = PROCESSED_DIR / f"bodhidharma_subset_{SUBSET_SIZE}_notes.csv"
ML_SUMMARY_CSV = PROCESSED_DIR / "ml_classifier_comparison_summary.csv"
ML_FEATURES_CSV = PROCESSED_DIR / "ml_song_features.csv"

CURVE_COLUMNS = ["time_norm", "pitch_norm", "velocity_norm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare distance methods and ML classifiers for genre recognition.")
    parser.add_argument(
        "--sequence-points",
        type=int,
        default=DEFAULT_SEQUENCE_POINTS,
        help="Number of melody curve points used by Hausdorff, DTW, and discrete Frechet.",
    )
    parser.add_argument(
        "--skip-ml",
        action="store_true",
        help="Only run distance methods. Useful when changing sequence-points because ML features do not depend on it.",
    )
    parser.add_argument(
        "--methods",
        default="Hausdorff,DTW,Discrete Frechet",
        help="Comma-separated distance methods to run: Hausdorff, DTW, Discrete Frechet.",
    )
    return parser.parse_args()


def distance_summary_path(sequence_points: int) -> Path:
    return PROCESSED_DIR / f"distance_method_comparison_summary_{sequence_points}.csv"


def distance_predictions_path(sequence_points: int) -> Path:
    return PROCESSED_DIR / f"distance_method_comparison_predictions_{sequence_points}.csv"


def accuracy_figure_path(sequence_points: int) -> Path:
    return FIGURES_DIR / f"method_comparison_accuracy_{sequence_points}.png"


def evenly_resample_array(values: np.ndarray, target_count: int) -> np.ndarray:
    if len(values) <= target_count:
        return values.copy()
    indices = np.linspace(0, len(values) - 1, target_count).round().astype(int)
    return values[indices]


def load_sequence_curves(sequence_points: int) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
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
        curve = group[CURVE_COLUMNS].to_numpy(float)
        curves[int(song_id)] = np.ascontiguousarray(evenly_resample_array(curve, sequence_points))
    return songs, curves


def directed_hausdorff_kdtree(a: np.ndarray, b: np.ndarray) -> float:
    tree = cKDTree(b)
    distances, _ = tree.query(a, k=1, workers=-1)
    return float(distances.max())


def hausdorff_distance(a: np.ndarray, b: np.ndarray) -> float:
    return max(directed_hausdorff_kdtree(a, b), directed_hausdorff_kdtree(b, a))


if njit is not None:

    @njit(cache=True)
    def point_distance(a: np.ndarray, b: np.ndarray, i: int, j: int) -> float:
        total = 0.0
        for k in range(a.shape[1]):
            diff = a[i, k] - b[j, k]
            total += diff * diff
        return np.sqrt(total)


    @njit(cache=True)
    def dtw_distance_numba(a: np.ndarray, b: np.ndarray) -> float:
        rows = a.shape[0]
        cols = b.shape[0]
        dp = np.full((rows + 1, cols + 1), np.inf)
        dp[0, 0] = 0.0
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                previous = dp[i - 1, j]
                if dp[i, j - 1] < previous:
                    previous = dp[i, j - 1]
                if dp[i - 1, j - 1] < previous:
                    previous = dp[i - 1, j - 1]
                dp[i, j] = point_distance(a, b, i - 1, j - 1) + previous
        return dp[rows, cols] / (rows + cols)


    @njit(cache=True)
    def discrete_frechet_distance_numba(a: np.ndarray, b: np.ndarray) -> float:
        rows = a.shape[0]
        cols = b.shape[0]
        ca = np.empty((rows, cols), dtype=np.float64)
        ca[0, 0] = point_distance(a, b, 0, 0)
        for i in range(1, rows):
            distance = point_distance(a, b, i, 0)
            ca[i, 0] = ca[i - 1, 0] if ca[i - 1, 0] > distance else distance
        for j in range(1, cols):
            distance = point_distance(a, b, 0, j)
            ca[0, j] = ca[0, j - 1] if ca[0, j - 1] > distance else distance
        for i in range(1, rows):
            for j in range(1, cols):
                previous = ca[i - 1, j]
                if ca[i - 1, j - 1] < previous:
                    previous = ca[i - 1, j - 1]
                if ca[i, j - 1] < previous:
                    previous = ca[i, j - 1]
                distance = point_distance(a, b, i, j)
                ca[i, j] = previous if previous > distance else distance
        return ca[rows - 1, cols - 1]


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    if njit is not None:
        return float(dtw_distance_numba(a, b))
    rows, cols = a.shape[0], b.shape[0]
    dp = np.full((rows + 1, cols + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            distance = np.linalg.norm(a[i - 1] - b[j - 1])
            dp[i, j] = distance + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[rows, cols] / (rows + cols))


def discrete_frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    if njit is not None:
        return float(discrete_frechet_distance_numba(a, b))
    rows, cols = a.shape[0], b.shape[0]
    ca = np.empty((rows, cols), dtype=float)
    ca[0, 0] = np.linalg.norm(a[0] - b[0])
    for i in range(1, rows):
        ca[i, 0] = max(ca[i - 1, 0], np.linalg.norm(a[i] - b[0]))
    for j in range(1, cols):
        ca[0, j] = max(ca[0, j - 1], np.linalg.norm(a[0] - b[j]))
    for i in range(1, rows):
        for j in range(1, cols):
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), np.linalg.norm(a[i] - b[j]))
    return float(ca[-1, -1])


def compute_distance_matrix(
    songs: pd.DataFrame,
    curves: dict[int, np.ndarray],
    distance_fn,
) -> np.ndarray:
    song_ids = songs["song_id"].astype(int).to_list()
    matrix = np.zeros((len(song_ids), len(song_ids)), dtype=float)
    for i, song_id_i in enumerate(song_ids):
        curve_i = curves[song_id_i]
        for j in range(i + 1, len(song_ids)):
            distance = distance_fn(curve_i, curves[song_ids[j]])
            matrix[i, j] = distance
            matrix[j, i] = distance
    return matrix


def evaluate_1nn(songs: pd.DataFrame, matrix: np.ndarray, method: str) -> pd.DataFrame:
    rows = []
    for i, song in songs.iterrows():
        distances = matrix[i].copy()
        distances[i] = np.inf
        nearest_idx = int(distances.argmin())
        nearest = songs.iloc[nearest_idx]
        rows.append(
            {
                "method": method,
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


def summarize_distance_matrix(
    songs: pd.DataFrame,
    matrix: np.ndarray,
    method: str,
    elapsed_seconds: float,
    sequence_points: int,
) -> dict[str, float]:
    genres = songs["genre"].to_numpy()
    upper_i, upper_j = np.triu_indices(len(songs), k=1)
    distances = matrix[upper_i, upper_j]
    same_mask = genres[upper_i] == genres[upper_j]
    same = distances[same_mask]
    different = distances[~same_mask]
    predictions = evaluate_1nn(songs, matrix, method)
    return {
        "method": method,
        "sequence_points": sequence_points,
        "accuracy": float(predictions["correct"].mean()),
        "same_genre_mean": float(same.mean()),
        "different_genre_mean": float(different.mean()),
        "mean_gap": float(different.mean() - same.mean()),
        "same_genre_median": float(np.median(same)),
        "different_genre_median": float(np.median(different)),
        "median_gap": float(np.median(different) - np.median(same)),
        "elapsed_seconds": elapsed_seconds,
    }


def entropy(values: np.ndarray) -> float:
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    probabilities = values / values.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def ratio(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def build_song_features() -> pd.DataFrame:
    notes = pd.read_csv(NOTES_CSV)
    notes = notes[notes["channel"].astype(int) != 9].copy()
    notes = notes[notes["duration_beat"] > 0].copy()

    rows = []
    for song_id, group in notes.sort_values(["song_id", "onset_beat", "pitch"]).groupby("song_id"):
        melody = group.sort_values(["onset_beat", "pitch"])
        pitches = melody["pitch"].to_numpy(float)
        velocities = melody["velocity"].to_numpy(float)
        durations = melody["duration_beat"].to_numpy(float)
        onsets = melody["onset_beat"].to_numpy(float)
        intervals = np.diff(pitches)
        onset_deltas = np.diff(np.unique(onsets))
        total_beats = max(float(melody["end_tick"].max() / melody["ticks_per_beat"].iloc[0]), 1.0)

        pitch_classes = np.bincount((pitches.astype(int) % 12), minlength=12).astype(float)
        duration_bins = np.histogram(durations, bins=[0, 0.25, 0.5, 1, 2, 4, np.inf])[0].astype(float)
        interval_abs = np.abs(intervals) if len(intervals) else np.array([0.0])

        row: dict[str, object] = {
            "song_id": int(song_id),
            "genre": melody["genre"].iloc[0],
            "file_name": melody["file_name"].iloc[0],
            "note_count": int(len(melody)),
            "track_count": int(melody["track"].nunique()),
            "channel_count": int(melody["channel"].nunique()),
            "total_beats": total_beats,
            "note_density": float(len(melody) / total_beats),
            "pitch_mean": float(pitches.mean()),
            "pitch_std": float(pitches.std()),
            "pitch_min": float(pitches.min()),
            "pitch_max": float(pitches.max()),
            "pitch_range": float(pitches.max() - pitches.min()),
            "velocity_mean": float(velocities.mean()),
            "velocity_std": float(velocities.std()),
            "duration_mean": float(durations.mean()),
            "duration_std": float(durations.std()),
            "duration_median": float(np.median(durations)),
            "duration_entropy": entropy(duration_bins),
            "pitch_class_entropy": entropy(pitch_classes),
            "interval_mean": float(intervals.mean()) if len(intervals) else 0.0,
            "interval_std": float(intervals.std()) if len(intervals) else 0.0,
            "interval_abs_mean": float(interval_abs.mean()),
            "interval_abs_std": float(interval_abs.std()),
            "ascending_ratio": ratio(int((intervals > 0).sum()), len(intervals)),
            "descending_ratio": ratio(int((intervals < 0).sum()), len(intervals)),
            "repeat_pitch_ratio": ratio(int((intervals == 0).sum()), len(intervals)),
            "onset_delta_mean": float(onset_deltas.mean()) if len(onset_deltas) else 0.0,
            "onset_delta_std": float(onset_deltas.std()) if len(onset_deltas) else 0.0,
        }

        for pitch_class, value in enumerate(pitch_classes / max(pitch_classes.sum(), 1.0)):
            row[f"pitch_class_{pitch_class}"] = float(value)
        for bin_index, value in enumerate(duration_bins / max(duration_bins.sum(), 1.0)):
            row[f"duration_bin_{bin_index}"] = float(value)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["genre", "file_name"]).reset_index(drop=True)


def evaluate_ml_classifiers(features: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [column for column in features.columns if column not in {"song_id", "genre", "file_name"}]
    x = features[feature_columns].to_numpy(float)
    y = LabelEncoder().fit_transform(features["genre"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    classifiers = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42),
        ),
        "SVM RBF": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=5.0, gamma="scale", class_weight="balanced", random_state=42),
        ),
        "kNN Features": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    rows = []
    for name, classifier in classifiers.items():
        start = time.perf_counter()
        scores = cross_val_score(classifier, x, y, cv=cv, scoring="accuracy", n_jobs=None)
        rows.append(
            {
                "method": name,
                "feature_count": len(feature_columns),
                "folds": cv.get_n_splits(),
                "accuracy_mean": float(scores.mean()),
                "accuracy_std": float(scores.std()),
                "fold_scores": ";".join(f"{score:.4f}" for score in scores),
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        print(f"{name}: accuracy={scores.mean():.4f} +/- {scores.std():.4f}")
    return pd.DataFrame(rows)


def plot_accuracy(distance_summary: pd.DataFrame, ml_summary: pd.DataFrame, output_path: Path) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    distance_plot = distance_summary[["method", "accuracy"]].rename(columns={"accuracy": "accuracy_mean"})
    distance_plot["family"] = "Distance 1-NN"
    ml_plot = ml_summary[["method", "accuracy_mean"]].copy()
    ml_plot["family"] = "ML classifier"
    plot_df = pd.concat([distance_plot, ml_plot], ignore_index=True)
    plot_df = plot_df.sort_values("accuracy_mean", ascending=True)

    colors = np.where(plot_df["family"] == "ML classifier", "#2ec4b6", "#ff9f1c")
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["method"], plot_df["accuracy_mean"], color=colors)
    plt.xlabel("Accuracy")
    plt.title("Distance Methods vs Machine Learning Classifiers")
    plt.xlim(0, max(0.65, plot_df["accuracy_mean"].max() + 0.08))
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    sequence_points = args.sequence_points
    songs, curves = load_sequence_curves(sequence_points)
    distance_methods = {
        "Hausdorff": hausdorff_distance,
        "DTW": dtw_distance,
        "Discrete Frechet": discrete_frechet_distance,
    }
    selected_methods = {method.strip() for method in args.methods.split(",") if method.strip()}
    unknown_methods = selected_methods - set(distance_methods)
    if unknown_methods:
        raise ValueError(f"Unknown distance methods: {sorted(unknown_methods)}")

    distance_summaries = []
    distance_predictions = []
    for method, distance_fn in distance_methods.items():
        if method not in selected_methods:
            continue
        start = time.perf_counter()
        matrix = compute_distance_matrix(songs, curves, distance_fn)
        elapsed = time.perf_counter() - start
        predictions = evaluate_1nn(songs, matrix, method)
        summary = summarize_distance_matrix(songs, matrix, method, elapsed, sequence_points)
        distance_summaries.append(summary)
        distance_predictions.append(predictions)
        print(
            f"{method}: accuracy={summary['accuracy']:.4f}, "
            f"same_mean={summary['same_genre_mean']:.4f}, "
            f"diff_mean={summary['different_genre_mean']:.4f}, "
            f"gap={summary['mean_gap']:.4f}, seconds={elapsed:.1f}"
        )

    distance_summary_df = pd.DataFrame(distance_summaries)
    distance_predictions_df = pd.concat(distance_predictions, ignore_index=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    distance_summary_csv = distance_summary_path(sequence_points)
    distance_predictions_csv = distance_predictions_path(sequence_points)
    distance_summary_df.to_csv(distance_summary_csv, index=False, encoding="utf-8")
    distance_predictions_df.to_csv(distance_predictions_csv, index=False, encoding="utf-8")

    if args.skip_ml and ML_SUMMARY_CSV.exists():
        ml_summary_df = pd.read_csv(ML_SUMMARY_CSV)
    else:
        song_features = build_song_features()
        ml_summary_df = evaluate_ml_classifiers(song_features)
        song_features.to_csv(ML_FEATURES_CSV, index=False, encoding="utf-8")
        ml_summary_df.to_csv(ML_SUMMARY_CSV, index=False, encoding="utf-8")

    method_accuracy_png = accuracy_figure_path(sequence_points)
    plot_accuracy(distance_summary_df, ml_summary_df, method_accuracy_png)

    print(f"distance summary: {distance_summary_csv}")
    print(f"distance predictions: {distance_predictions_csv}")
    print(f"ml features: {ML_FEATURES_CSV}")
    print(f"ml summary: {ML_SUMMARY_CSV}")
    print(f"figure: {method_accuracy_png}")


if __name__ == "__main__":
    main()
