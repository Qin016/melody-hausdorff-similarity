from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from extract_melody_points import build_melody_points, parse_midi
from method_comparison_experiment import discrete_frechet_distance, dtw_distance, hausdorff_distance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_lakh"
FIGURES_DIR = PROJECT_ROOT / "figures" / "lakh"

SUBSET_CSV = PROCESSED_DIR / "lakh_tagtraum_balanced_subset.csv"
FEATURES_CSV = PROCESSED_DIR / "lakh_enhanced_song_features.csv"
POINTS_CSV = PROCESSED_DIR / "lakh_melody_points_sampled_300.csv"
ML_SUMMARY_CSV = PROCESSED_DIR / "lakh_ml_classifier_summary.csv"
DISTANCE_SUMMARY_CSV = PROCESSED_DIR / "lakh_distance_method_summary.csv"
ML_ACCURACY_PNG = FIGURES_DIR / "lakh_ml_accuracy.png"
DISTANCE_ACCURACY_PNG = FIGURES_DIR / "lakh_distance_accuracy.png"

MAX_POINTS_PER_SONG = 300
DISTANCE_PER_GENRE = 20
CURVE_COLUMNS = ["time_norm", "pitch_norm", "velocity_norm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run genre experiments on the Lakh/tagtraum MIDI subset.")
    parser.add_argument("--distance-per-genre", type=int, default=DISTANCE_PER_GENRE)
    parser.add_argument("--skip-distance", action="store_true")
    parser.add_argument("--skip-parse", action="store_true", help="Reuse existing feature and point CSV files.")
    return parser.parse_args()


def entropy(values: np.ndarray) -> float:
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    probabilities = values / values.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def ratio(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def evenly_sample_points(points: list[dict[str, object]], max_points: int) -> list[dict[str, object]]:
    if len(points) <= max_points:
        sampled = points
    else:
        indices = np.linspace(0, len(points) - 1, max_points).round().astype(int)
        sampled = [points[int(index)] for index in indices]
    output = []
    for sample_index, point in enumerate(sampled, start=1):
        output.append({**point, "sample_index": sample_index, "sample_count": len(sampled)})
    return output


def feature_row(notes: list[dict[str, int]], ticks_per_beat: int, meta: dict[str, str]) -> dict[str, object] | None:
    pitched = [note for note in notes if int(note["channel"]) != 9 and int(note["duration_tick"]) > 0]
    percussion = [note for note in notes if int(note["channel"]) == 9 and int(note["duration_tick"]) >= 0]
    if len(pitched) < 10:
        return None

    pitches = np.array([note["pitch"] for note in pitched], dtype=float)
    velocities = np.array([note["velocity"] for note in pitched], dtype=float)
    durations = np.array([note["duration_tick"] / ticks_per_beat for note in pitched], dtype=float)
    onsets = np.array([note["onset_tick"] / ticks_per_beat for note in pitched], dtype=float)
    intervals = np.diff(pitches)
    unique_onsets, onset_counts = np.unique(onsets, return_counts=True)
    onset_deltas = np.diff(unique_onsets)
    end_beat = max(note["end_tick"] for note in notes) / ticks_per_beat
    total_beats = max(float(end_beat), 1.0)

    pitch_classes = np.bincount((pitches.astype(int) % 12), minlength=12).astype(float)
    duration_bins = np.histogram(durations, bins=[0, 0.25, 0.5, 1, 2, 4, np.inf])[0].astype(float)
    interval_bins = np.histogram(np.clip(intervals, -12, 12), bins=np.arange(-12.5, 13.5, 1))[0].astype(float)
    beat_positions = np.mod(onsets, 4.0)
    beat_bins = np.histogram(beat_positions, bins=np.linspace(0, 4, 17))[0].astype(float)
    percussion_pitches = np.array([note["pitch"] for note in percussion], dtype=int) if percussion else np.array([], dtype=int)

    row: dict[str, object] = {
        "song_id": int(meta["song_id"]),
        "track_id": meta["track_id"],
        "genre": meta["genre"],
        "file_name": meta["file_name"],
        "note_count": len(pitched),
        "percussion_count": len(percussion),
        "track_count": len({note["track"] for note in notes}),
        "channel_count": len({note["channel"] for note in notes}),
        "total_beats": total_beats,
        "note_density": len(pitched) / total_beats,
        "percussion_density": len(percussion) / total_beats,
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
        "interval_abs_mean": float(np.abs(intervals).mean()) if len(intervals) else 0.0,
        "interval_abs_std": float(np.abs(intervals).std()) if len(intervals) else 0.0,
        "ascending_ratio": ratio(int((intervals > 0).sum()), len(intervals)),
        "descending_ratio": ratio(int((intervals < 0).sum()), len(intervals)),
        "repeat_pitch_ratio": ratio(int((intervals == 0).sum()), len(intervals)),
        "onset_delta_mean": float(onset_deltas.mean()) if len(onset_deltas) else 0.0,
        "onset_delta_std": float(onset_deltas.std()) if len(onset_deltas) else 0.0,
        "polyphony_mean": float(onset_counts.mean()),
        "polyphony_max": float(onset_counts.max()),
        "polyphonic_onset_ratio": ratio(int((onset_counts > 1).sum()), len(onset_counts)),
        "kick_ratio": ratio(int(np.isin(percussion_pitches, [35, 36]).sum()), len(percussion_pitches)),
        "snare_ratio": ratio(int(np.isin(percussion_pitches, [38, 40]).sum()), len(percussion_pitches)),
        "hihat_ratio": ratio(int(np.isin(percussion_pitches, [42, 44, 46]).sum()), len(percussion_pitches)),
    }

    for pitch_class, value in enumerate(pitch_classes / max(pitch_classes.sum(), 1.0)):
        row[f"pitch_class_{pitch_class}"] = float(value)
    for bin_index, value in enumerate(duration_bins / max(duration_bins.sum(), 1.0)):
        row[f"duration_bin_{bin_index}"] = float(value)
    for bin_index, value in enumerate(interval_bins / max(interval_bins.sum(), 1.0)):
        row[f"interval_bin_{bin_index}"] = float(value)
    for bin_index, value in enumerate(beat_bins / max(beat_bins.sum(), 1.0)):
        row[f"beat_bin_{bin_index}"] = float(value)
    return row


def parse_lakh_midis() -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = pd.read_csv(SUBSET_CSV)
    feature_rows = []
    point_rows = []
    failures = []

    for index, meta in subset.iterrows():
        midi_path = PROJECT_ROOT / meta["relative_path"]
        try:
            ticks_per_beat, notes = parse_midi(midi_path)
        except Exception as exc:
            failures.append(f"{meta['relative_path']}: {exc}")
            continue

        row = feature_row(notes, ticks_per_beat, meta.to_dict())
        if row is None:
            failures.append(f"{meta['relative_path']}: too few pitched notes")
            continue
        feature_rows.append(row)

        melody_points = build_melody_points(notes)
        sampled = evenly_sample_points(melody_points, MAX_POINTS_PER_SONG)
        for point in sampled:
            point_rows.append(
                {
                    "song_id": int(meta["song_id"]),
                    "track_id": meta["track_id"],
                    "genre": meta["genre"],
                    "file_name": meta["file_name"],
                    "relative_path": meta["relative_path"],
                    **point,
                }
            )

        if (index + 1) % 100 == 0:
            print(f"parsed {index + 1}/{len(subset)} MIDI files")

    features = pd.DataFrame(feature_rows).sort_values(["genre", "file_name"]).reset_index(drop=True)
    points = pd.DataFrame(point_rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_CSV, index=False, encoding="utf-8")
    points.to_csv(POINTS_CSV, index=False, encoding="utf-8")
    if failures:
        (PROCESSED_DIR / "lakh_parse_failures.txt").write_text("\n".join(failures), encoding="utf-8")
    print(f"usable songs: {len(features)}")
    print(f"features: {FEATURES_CSV}")
    print(f"points: {POINTS_CSV}")
    print(f"failures: {len(failures)}")
    return features, points


def evaluate_ml(features: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [column for column in features.columns if column not in {"song_id", "track_id", "genre", "file_name"}]
    x = features[feature_columns].fillna(0).to_numpy(float)
    y = LabelEncoder().fit_transform(features["genre"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", random_state=42))
    rf = RandomForestClassifier(n_estimators=600, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
    et = ExtraTreesClassifier(n_estimators=800, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=2.0, class_weight="balanced", random_state=42))
    vote = VotingClassifier(
        estimators=[("svm", svm), ("rf", rf), ("et", et), ("lr", lr)],
        voting="hard",
    )
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
        scores = cross_val_score(classifier, x, y, cv=cv, scoring="accuracy")
        rows.append(
            {
                "method": name,
                "songs": len(features),
                "genres": features["genre"].nunique(),
                "feature_count": len(feature_columns),
                "accuracy_mean": float(scores.mean()),
                "accuracy_std": float(scores.std()),
                "fold_scores": ";".join(f"{score:.4f}" for score in scores),
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        print(f"{name}: {scores.mean():.4f} +/- {scores.std():.4f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(ML_SUMMARY_CSV, index=False, encoding="utf-8")
    plot_accuracy(summary, "accuracy_mean", "Lakh/tagtraum ML Classifier Accuracy", ML_ACCURACY_PNG)
    return summary


def load_distance_curves(points: pd.DataFrame, per_genre: int) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    songs = (
        points[["song_id", "track_id", "genre", "file_name", "relative_path"]]
        .drop_duplicates()
        .sort_values(["genre", "file_name"])
        .groupby("genre", group_keys=False)
        .head(per_genre)
        .reset_index(drop=True)
    )
    keep_ids = set(songs["song_id"].astype(int))
    curves: dict[int, np.ndarray] = {}
    for song_id, group in points[points["song_id"].isin(keep_ids)].sort_values(["song_id", "sample_index"]).groupby("song_id"):
        curves[int(song_id)] = np.ascontiguousarray(group[CURVE_COLUMNS].to_numpy(float))
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


def one_nn_accuracy(songs: pd.DataFrame, matrix: np.ndarray) -> float:
    genres = songs["genre"].to_numpy()
    correct = 0
    for i in range(len(songs)):
        row = matrix[i].copy()
        row[i] = np.inf
        nearest_idx = int(row.argmin())
        correct += int(genres[i] == genres[nearest_idx])
    return correct / len(songs)


def evaluate_distances(points: pd.DataFrame, per_genre: int) -> pd.DataFrame:
    songs, curves = load_distance_curves(points, per_genre)
    methods = {
        "Hausdorff": hausdorff_distance,
        "DTW": dtw_distance,
        "Discrete Frechet": discrete_frechet_distance,
    }
    rows = []
    for name, distance_fn in methods.items():
        start = time.perf_counter()
        matrix = compute_distance_matrix(songs, curves, distance_fn)
        elapsed = time.perf_counter() - start
        accuracy = one_nn_accuracy(songs, matrix)
        genres = songs["genre"].to_numpy()
        upper_i, upper_j = np.triu_indices(len(songs), k=1)
        distances = matrix[upper_i, upper_j]
        same = distances[genres[upper_i] == genres[upper_j]]
        different = distances[genres[upper_i] != genres[upper_j]]
        rows.append(
            {
                "method": name,
                "songs": len(songs),
                "genres": songs["genre"].nunique(),
                "per_genre": per_genre,
                "accuracy": accuracy,
                "same_genre_mean": float(same.mean()),
                "different_genre_mean": float(different.mean()),
                "mean_gap": float(different.mean() - same.mean()),
                "elapsed_seconds": elapsed,
            }
        )
        print(f"{name}: accuracy={accuracy:.4f}, seconds={elapsed:.1f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(DISTANCE_SUMMARY_CSV, index=False, encoding="utf-8")
    plot_accuracy(summary, "accuracy", "Lakh/tagtraum Distance 1-NN Accuracy", DISTANCE_ACCURACY_PNG)
    return summary


def plot_accuracy(summary: pd.DataFrame, value_column: str, title: str, output_path: Path) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = summary.sort_values(value_column)
    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["method"], plot_df[value_column], color="#2ec4b6")
    plt.xlabel("Accuracy")
    plt.title(title)
    plt.xlim(0, max(0.75, float(plot_df[value_column].max()) + 0.08))
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
        features, points = parse_lakh_midis()

    ml_summary = evaluate_ml(features)
    print(f"ml summary: {ML_SUMMARY_CSV}")
    if not args.skip_distance:
        distance_summary = evaluate_distances(points, args.distance_per_genre)
        print(f"distance summary: {DISTANCE_SUMMARY_CSV}")
    print(ml_summary.to_string(index=False))


if __name__ == "__main__":
    main()
