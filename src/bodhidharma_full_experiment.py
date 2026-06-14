from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from extract_melody_points import build_melody_points, parse_midi
from lakh_genre_experiment import evenly_sample_points, feature_row
from method_comparison_experiment import discrete_frechet_distance, dtw_distance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

METADATA_CSV = PROCESSED_DIR / "bodhidharma_metadata.csv"
FEATURES_CSV = PROCESSED_DIR / "bodhidharma_full_enhanced_song_features.csv"
POINTS_CSV = PROCESSED_DIR / "bodhidharma_full_melody_points_sampled_300.csv"
ML_SUMMARY_CSV = PROCESSED_DIR / "bodhidharma_full_ml_classifier_summary.csv"
DISTANCE_SUMMARY_CSV = PROCESSED_DIR / "bodhidharma_full_distance_method_summary.csv"
PARSE_FAILURES_TXT = PROCESSED_DIR / "bodhidharma_full_parse_failures.txt"
ML_ACCURACY_PNG = FIGURES_DIR / "bodhidharma_full_ml_accuracy.png"
DISTANCE_ACCURACY_PNG = FIGURES_DIR / "bodhidharma_full_distance_accuracy.png"

MAX_POINTS_PER_SONG = 300
DEFAULT_DISTANCE_POINTS = 100
CURVE_COLUMNS = ["time_norm", "pitch_norm", "velocity_norm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full, unbalanced Bodhidharma genre experiments.")
    parser.add_argument("--skip-parse", action="store_true")
    parser.add_argument("--skip-distance", action="store_true")
    parser.add_argument("--distance-points", type=int, default=DEFAULT_DISTANCE_POINTS)
    return parser.parse_args()


@njit(cache=True)
def directed_hausdorff_squared(a: np.ndarray, b: np.ndarray) -> float:
    max_min = 0.0
    for i in range(a.shape[0]):
        best = np.inf
        for j in range(b.shape[0]):
            total = 0.0
            for k in range(a.shape[1]):
                diff = a[i, k] - b[j, k]
                total += diff * diff
            if total < best:
                best = total
        if best > max_min:
            max_min = best
    return max_min


@njit(cache=True)
def hausdorff_distance_numba(a: np.ndarray, b: np.ndarray) -> float:
    ab = directed_hausdorff_squared(a, b)
    ba = directed_hausdorff_squared(b, a)
    return np.sqrt(ab if ab > ba else ba)


def evenly_resample_array(values: np.ndarray, target_count: int) -> np.ndarray:
    if len(values) <= target_count:
        return values.copy()
    indices = np.linspace(0, len(values) - 1, target_count).round().astype(int)
    return values[indices]


def parse_full_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(METADATA_CSV)
    metadata = metadata[metadata["parse_status"] == "ok"].sort_values(["genre", "file_name"]).reset_index(drop=True)
    feature_rows = []
    point_rows = []
    failures = []

    for index, row in metadata.iterrows():
        song_id = index + 1
        meta = {
            "song_id": song_id,
            "track_id": Path(row["relative_path"]).stem,
            "genre": row["genre"],
            "file_name": row["file_name"],
        }
        midi_path = PROJECT_ROOT / row["relative_path"]
        try:
            ticks_per_beat, notes = parse_midi(midi_path)
        except Exception as exc:
            failures.append(f"{row['relative_path']}: {exc}")
            continue

        features = feature_row(notes, ticks_per_beat, meta)
        if features is None:
            failures.append(f"{row['relative_path']}: too few pitched notes")
            continue
        feature_rows.append(features)

        melody_points = build_melody_points(notes)
        sampled = evenly_sample_points(melody_points, MAX_POINTS_PER_SONG)
        for point in sampled:
            point_rows.append(
                {
                    "song_id": song_id,
                    "genre": row["genre"],
                    "file_name": row["file_name"],
                    "relative_path": row["relative_path"],
                    **point,
                }
            )

        if (index + 1) % 100 == 0:
            print(f"parsed {index + 1}/{len(metadata)} MIDI files")

    features_df = pd.DataFrame(feature_rows).sort_values(["genre", "file_name"]).reset_index(drop=True)
    points_df = pd.DataFrame(point_rows)
    features_df.to_csv(FEATURES_CSV, index=False, encoding="utf-8")
    points_df.to_csv(POINTS_CSV, index=False, encoding="utf-8")
    PARSE_FAILURES_TXT.write_text("\n".join(failures), encoding="utf-8")
    print(f"usable songs: {len(features_df)}")
    print(f"failures: {len(failures)}")
    print(f"features: {FEATURES_CSV}")
    print(f"points: {POINTS_CSV}")
    return features_df, points_df


def evaluate_ml(features: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [column for column in features.columns if column not in {"song_id", "track_id", "genre", "file_name"}]
    x = features[feature_columns].fillna(0).to_numpy(float)
    encoder = LabelEncoder()
    y = encoder.fit_transform(features["genre"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", random_state=42))
    rf = RandomForestClassifier(n_estimators=600, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
    et = ExtraTreesClassifier(n_estimators=800, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=2.0, class_weight="balanced", random_state=42))
    vote = VotingClassifier(estimators=[("svm", svm), ("rf", rf), ("et", et), ("lr", lr)], voting="hard")

    classifiers = {
        "Logistic Regression": lr,
        "SVM RBF": svm,
        "Random Forest": rf,
        "Extra Trees": et,
        "Voting Ensemble": vote,
    }
    rows = []
    for name, classifier in classifiers.items():
        start = time.perf_counter()
        accuracy = cross_val_score(classifier, x, y, cv=cv, scoring="accuracy")
        balanced = cross_val_score(classifier, x, y, cv=cv, scoring="balanced_accuracy")
        rows.append(
            {
                "method": name,
                "songs": len(features),
                "genres": features["genre"].nunique(),
                "feature_count": len(feature_columns),
                "accuracy_mean": float(accuracy.mean()),
                "accuracy_std": float(accuracy.std()),
                "balanced_accuracy_mean": float(balanced.mean()),
                "balanced_accuracy_std": float(balanced.std()),
                "accuracy_fold_scores": ";".join(f"{score:.4f}" for score in accuracy),
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        print(f"{name}: accuracy={accuracy.mean():.4f}, balanced={balanced.mean():.4f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(ML_SUMMARY_CSV, index=False, encoding="utf-8")
    plot_accuracy(summary, "balanced_accuracy_mean", "Full Bodhidharma ML Balanced Accuracy", ML_ACCURACY_PNG)
    return summary


def load_curves(points: pd.DataFrame, distance_points: int) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    songs = (
        points[["song_id", "genre", "file_name", "relative_path"]]
        .drop_duplicates()
        .sort_values(["genre", "file_name"])
        .reset_index(drop=True)
    )
    curves: dict[int, np.ndarray] = {}
    for song_id, group in points.sort_values(["song_id", "sample_index"]).groupby("song_id"):
        curve = group[CURVE_COLUMNS].to_numpy(float)
        curves[int(song_id)] = np.ascontiguousarray(evenly_resample_array(curve, distance_points))
    return songs, curves


def compute_distance_matrix(songs: pd.DataFrame, curves: dict[int, np.ndarray], distance_fn) -> np.ndarray:
    song_ids = songs["song_id"].astype(int).to_list()
    matrix = np.zeros((len(song_ids), len(song_ids)), dtype=float)
    for i, song_id_i in enumerate(song_ids):
        for j in range(i + 1, len(song_ids)):
            distance = distance_fn(curves[song_id_i], curves[song_ids[j]])
            matrix[i, j] = distance
            matrix[j, i] = distance
    return matrix


def distance_scores(songs: pd.DataFrame, matrix: np.ndarray) -> tuple[float, float]:
    genres = songs["genre"].to_numpy()
    predictions = []
    for i in range(len(songs)):
        row = matrix[i].copy()
        row[i] = np.inf
        predictions.append(genres[int(row.argmin())])
    return float(np.mean(genres == np.array(predictions))), float(balanced_accuracy_score(genres, predictions))


def evaluate_distances(points: pd.DataFrame, distance_points: int) -> pd.DataFrame:
    songs, curves = load_curves(points, distance_points)
    methods = {
        "Hausdorff": lambda a, b: float(hausdorff_distance_numba(a, b)),
        "DTW": dtw_distance,
        "Discrete Frechet": discrete_frechet_distance,
    }
    rows = []
    genres = songs["genre"].to_numpy()
    upper_i, upper_j = np.triu_indices(len(songs), k=1)
    for name, distance_fn in methods.items():
        start = time.perf_counter()
        matrix = compute_distance_matrix(songs, curves, distance_fn)
        elapsed = time.perf_counter() - start
        accuracy, balanced = distance_scores(songs, matrix)
        distances = matrix[upper_i, upper_j]
        same = distances[genres[upper_i] == genres[upper_j]]
        different = distances[genres[upper_i] != genres[upper_j]]
        rows.append(
            {
                "method": name,
                "songs": len(songs),
                "genres": songs["genre"].nunique(),
                "distance_points": distance_points,
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "same_genre_mean": float(same.mean()),
                "different_genre_mean": float(different.mean()),
                "mean_gap": float(different.mean() - same.mean()),
                "elapsed_seconds": elapsed,
            }
        )
        print(f"{name}: accuracy={accuracy:.4f}, balanced={balanced:.4f}, seconds={elapsed:.1f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(DISTANCE_SUMMARY_CSV, index=False, encoding="utf-8")
    plot_accuracy(summary, "balanced_accuracy", "Full Bodhidharma Distance Balanced Accuracy", DISTANCE_ACCURACY_PNG)
    return summary


def plot_accuracy(summary: pd.DataFrame, column: str, title: str, output_path: Path) -> None:
    plot_df = summary.sort_values(column)
    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["method"], plot_df[column], color="#2ec4b6")
    plt.xlabel("Balanced accuracy")
    plt.title(title)
    plt.xlim(0, max(0.75, float(plot_df[column].max()) + 0.08))
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.skip_parse:
        features = pd.read_csv(FEATURES_CSV)
        points = pd.read_csv(POINTS_CSV)
    else:
        features, points = parse_full_dataset()

    ml_summary = evaluate_ml(features)
    print(f"ml summary: {ML_SUMMARY_CSV}")
    if not args.skip_distance:
        distance_summary = evaluate_distances(points, args.distance_points)
        print(f"distance summary: {DISTANCE_SUMMARY_CSV}")
    print(ml_summary.to_string(index=False))


if __name__ == "__main__":
    main()
