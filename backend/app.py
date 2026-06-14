from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
GENERATED_DIR = PROJECT_ROOT / "generated"
SRC_DIR = PROJECT_ROOT / "src"
BALANCED_POINTS_CSV = DATA_DIR / "bodhidharma_subset_25_melody_points_sampled_300.csv"
FULL_POINTS_CSV = DATA_DIR / "bodhidharma_full_melody_points_sampled_300.csv"

app = Flask(__name__)
CORS(app)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def artifact_url(path: Path) -> str:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    return f"/artifacts/{relative}"


def run_script(script: str, *args: str) -> None:
    command = [sys.executable, str(SRC_DIR / script), *args]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def dataset_key() -> str:
    value = request.args.get("dataset", default="balanced")
    return "full" if value == "full" else "balanced"


def points_csv_for(dataset: str) -> Path:
    return FULL_POINTS_CSV if dataset == "full" else BALANCED_POINTS_CSV


def resample_curve(curve: pd.DataFrame, max_points: int = 100) -> pd.DataFrame:
    curve = curve.sort_values("sample_index").reset_index(drop=True)
    if len(curve) <= max_points:
        return curve
    indices = np.linspace(0, len(curve) - 1, max_points).round().astype(int)
    return curve.iloc[indices].reset_index(drop=True)


def directed_hausdorff(a, b) -> float:
    distances = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2) ** 0.5
    return float(distances.min(axis=1).max())


def hausdorff(a, b) -> float:
    return max(directed_hausdorff(a, b), directed_hausdorff(b, a))


def plot_search_match(query_song_id: int, match_song_id: int, dataset: str = "balanced") -> Path:
    points = read_csv(points_csv_for(dataset))
    query = points[points["song_id"] == query_song_id].sort_values("sample_index")
    match = points[points["song_id"] == match_song_id].sort_values("sample_index")
    if query.empty or match.empty:
        return FIGURES_DIR / "similarity_search_top_match_3d.png"

    suffix = "full" if dataset == "full" else "balanced"
    output = FIGURES_DIR / f"similarity_search_top_match_3d_{suffix}.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 8), facecolor="#080b12")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#080b12")
    ax.plot(query["time_norm"], query["pitch_norm"], query["velocity_norm"], color="#ffcc00", linewidth=2.3, label="Query")
    ax.plot(
        match["time_norm"],
        match["pitch_norm"],
        match["velocity_norm"],
        color="#00e5ff",
        linewidth=2.3,
        label=f"Top match: {match['genre'].iloc[0]} / {match['file_name'].iloc[0]}",
    )
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor("#101522")
        axis.pane.set_edgecolor("#2a3040")
        axis._axinfo["grid"]["color"] = "#293241"
    ax.tick_params(colors="#d8dee9")
    ax.xaxis.label.set_color("#d8dee9")
    ax.yaxis.label.set_color("#d8dee9")
    ax.zaxis.label.set_color("#d8dee9")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pitch")
    ax.set_zlabel("Velocity")
    ax.set_title("3D Melody Similarity Search", color="#f8f9fa", pad=18)
    legend = ax.legend()
    legend.get_frame().set_facecolor("#101522")
    for text in legend.get_texts():
        text.set_color("#f8f9fa")
    ax.view_init(elev=24, azim=235)
    plt.tight_layout()
    plt.savefig(output, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


def song_table(points: pd.DataFrame) -> pd.DataFrame:
    return (
        points[["song_id", "genre", "file_name", "relative_path"]]
        .drop_duplicates()
        .sort_values(["genre", "file_name"])
        .reset_index(drop=True)
    )


def full_search_results(song_id: int, top_k: int) -> pd.DataFrame:
    points = read_csv(FULL_POINTS_CSV)
    songs_df = song_table(points).set_index("song_id")
    query = points[points["song_id"] == song_id]
    if query.empty:
        return pd.DataFrame()
    query_curve = resample_curve(query)[["time_norm", "pitch_norm", "velocity_norm"]].to_numpy(float)

    rows = []
    for candidate_id, group in points.groupby("song_id"):
        candidate_id = int(candidate_id)
        if candidate_id == song_id:
            continue
        candidate_curve = resample_curve(group)[["time_norm", "pitch_norm", "velocity_norm"]].to_numpy(float)
        rows.append(
            {
                "song_id": candidate_id,
                "genre": songs_df.loc[candidate_id, "genre"],
                "file_name": songs_df.loc[candidate_id, "file_name"],
                "relative_path": songs_df.loc[candidate_id, "relative_path"],
                "hausdorff_distance": hausdorff(query_curve, candidate_curve),
            }
        )
    return pd.DataFrame(rows).sort_values("hausdorff_distance").head(top_k).reset_index(drop=True)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "project_root": str(PROJECT_ROOT)})


@app.get("/api/summary")
def summary():
    dataset = dataset_key()
    label_summary = read_csv(DATA_DIR / "bodhidharma_label_summary.csv")
    points = read_csv(points_csv_for(dataset))

    artifacts = {
        "single_curve": artifact_url(FIGURES_DIR / "melody_curve_3d_single_neon.png"),
        "genre_curves": artifact_url(FIGURES_DIR / "melody_curve_3d_genre_comparison.png"),
        "interactive_curves": artifact_url(FIGURES_DIR / "interactive" / "melody_curves_3d_interactive.html"),
        "music_map": artifact_url(FIGURES_DIR / "music_map_mds.png"),
        "interactive_music_map": artifact_url(FIGURES_DIR / "interactive" / "music_map_mds_interactive.html"),
        "interpolation": artifact_url(FIGURES_DIR / "melody_interpolation_3d.png"),
        "generated_midi": artifact_url(GENERATED_DIR / "interpolated_melody.mid"),
    }
    if dataset == "full":
        distance_summary = read_csv(DATA_DIR / "bodhidharma_full_distance_method_summary.csv")
        hausdorff_row = distance_summary[distance_summary["method"] == "Hausdorff"].iloc[0]
        ml_summary = read_csv(DATA_DIR / "bodhidharma_full_ml_classifier_summary.csv")
        best_ml = ml_summary.sort_values("balanced_accuracy_mean", ascending=False).iloc[0]
        hausdorff_summary = {
            "pairwise_distances": int(points["song_id"].nunique() * (points["song_id"].nunique() - 1) / 2),
            "same_genre_mean": float(hausdorff_row["same_genre_mean"]),
            "different_genre_mean": float(hausdorff_row["different_genre_mean"]),
            "same_genre_median": None,
            "different_genre_median": None,
            "one_nn_accuracy": float(hausdorff_row["accuracy"]),
            "balanced_accuracy": float(hausdorff_row["balanced_accuracy"]),
            "best_ml_method": best_ml["method"],
            "best_ml_accuracy": float(best_ml["accuracy_mean"]),
            "best_ml_balanced_accuracy": float(best_ml["balanced_accuracy_mean"]),
        }
        artifacts.update(
            {
                "distance_heatmap": artifact_url(FIGURES_DIR / "bodhidharma_full_distance_accuracy.png"),
                "same_diff_boxplot": artifact_url(FIGURES_DIR / "bodhidharma_full_ml_accuracy.png"),
            }
        )
    else:
        pairwise = read_csv(DATA_DIR / "hausdorff_subset_25_pairwise_distances.csv")
        knn = read_csv(DATA_DIR / "hausdorff_subset_25_1nn_predictions.csv")
        same = pairwise[pairwise["same_genre"]]["hausdorff_distance"]
        diff = pairwise[~pairwise["same_genre"]]["hausdorff_distance"]
        hausdorff_summary = {
            "pairwise_distances": int(pairwise.shape[0]),
            "same_genre_mean": float(same.mean()),
            "different_genre_mean": float(diff.mean()),
            "same_genre_median": float(same.median()),
            "different_genre_median": float(diff.median()),
            "one_nn_accuracy": float(knn["correct"].mean()),
            "balanced_accuracy": None,
            "best_ml_method": None,
            "best_ml_accuracy": None,
            "best_ml_balanced_accuracy": None,
        }
        artifacts.update(
            {
                "distance_heatmap": artifact_url(FIGURES_DIR / "hausdorff_distance_heatmap.png"),
                "same_diff_boxplot": artifact_url(FIGURES_DIR / "hausdorff_same_vs_diff_boxplot.png"),
                "cluster": artifact_url(FIGURES_DIR / "hausdorff_hierarchical_clustering.png"),
            }
        )

    return jsonify(
        {
            "view": dataset,
            "dataset": {
                "midi_files": int(label_summary["file_count"].sum()),
                "genres": int(label_summary.shape[0]),
                "balanced_songs": int(points["song_id"].nunique()),
                "sampled_points": int(points.shape[0]),
                "label_summary": label_summary.to_dict(orient="records"),
            },
            "hausdorff": hausdorff_summary,
            "artifacts": artifacts,
        }
    )


@app.get("/api/songs")
def songs():
    points = read_csv(points_csv_for(dataset_key()))
    songs_df = song_table(points)
    return jsonify(songs_df.to_dict(orient="records"))


@app.get("/api/search")
def search():
    dataset = dataset_key()
    song_id = request.args.get("song_id", type=int)
    top_k = request.args.get("top_k", default=8, type=int)
    if song_id is None:
        return jsonify({"error": "song_id is required"}), 400

    if dataset == "full":
        results = full_search_results(song_id, top_k)
        if results.empty:
            return jsonify({"error": f"song_id {song_id} not found in full dataset"}), 404
    else:
        pairwise = read_csv(DATA_DIR / "hausdorff_subset_25_pairwise_distances.csv")
        songs_df = song_table(read_csv(BALANCED_POINTS_CSV)).set_index("song_id")

        rows = []
        for row in pairwise.itertuples(index=False):
            if int(row.song_id_a) == song_id:
                rows.append(
                    {
                        "song_id": int(row.song_id_b),
                        "genre": row.genre_b,
                        "file_name": row.file_name_b,
                        "relative_path": songs_df.loc[int(row.song_id_b), "relative_path"],
                        "hausdorff_distance": float(row.hausdorff_distance),
                    }
                )
            elif int(row.song_id_b) == song_id:
                rows.append(
                    {
                        "song_id": int(row.song_id_a),
                        "genre": row.genre_a,
                        "file_name": row.file_name_a,
                        "relative_path": songs_df.loc[int(row.song_id_a), "relative_path"],
                        "hausdorff_distance": float(row.hausdorff_distance),
                    }
                )

        if not rows:
            return jsonify({"error": f"song_id {song_id} not found in pairwise distances"}), 404
        results = pd.DataFrame(rows).sort_values("hausdorff_distance").head(top_k).reset_index(drop=True)

    results.insert(0, "rank", range(1, len(results) + 1))
    results.to_csv(DATA_DIR / f"similarity_search_results_{dataset}.csv", index=False, encoding="utf-8")
    figure = plot_search_match(song_id, int(results.iloc[0]["song_id"]), dataset)
    return jsonify(
        {
            "dataset": dataset,
            "query_song_id": song_id,
            "results": results.to_dict(orient="records"),
            "figure": artifact_url(figure),
        }
    )


@app.get("/api/music-map")
def music_map():
    map_csv = DATA_DIR / "music_map_mds.csv"
    if not map_csv.exists():
        run_script("music_map.py")
    data = read_csv(map_csv)
    return jsonify(
        {
            "points": data.to_dict(orient="records"),
            "figure": artifact_url(FIGURES_DIR / "music_map_mds.png"),
            "interactive": artifact_url(FIGURES_DIR / "interactive" / "music_map_mds_interactive.html"),
        }
    )


@app.post("/api/generate")
def generate():
    payload = request.get_json(silent=True) or {}
    alpha = float(payload.get("alpha", 0.5))
    note_count = int(payload.get("note_count", 180))
    min_distance = float(payload.get("min_distance", 0.22))
    smooth_window = int(payload.get("smooth_window", 9))
    song_a = payload.get("song_a")
    song_b = payload.get("song_b")
    if (song_a is None) != (song_b is None):
        return jsonify({"error": "song_a and song_b must be provided together"}), 400
    if song_a is not None and song_b is not None:
        try:
            song_a = int(song_a)
            song_b = int(song_b)
        except (TypeError, ValueError):
            return jsonify({"error": "song_a and song_b must be integers"}), 400
        if song_a == song_b:
            return jsonify({"error": "song_a and song_b must be different"}), 400

    args = [
        "--alpha",
        str(alpha),
        "--note-count",
        str(note_count),
        "--min-distance",
        str(min_distance),
        "--smooth-window",
        str(smooth_window),
    ]
    if song_a is not None and song_b is not None:
        args.extend(["--song-a", str(song_a), "--song-b", str(song_b)])

    run_script("melody_interpolation.py", *args)
    return jsonify(
        {
            "alpha": alpha,
            "note_count": note_count,
            "min_distance": min_distance,
            "smooth_window": smooth_window,
            "song_a": song_a,
            "song_b": song_b,
            "points_csv": artifact_url(DATA_DIR / "interpolated_melody_points.csv"),
            "midi": artifact_url(GENERATED_DIR / "interpolated_melody.mid"),
            "figure": artifact_url(FIGURES_DIR / "melody_interpolation_3d.png"),
        }
    )


@app.get("/artifacts/<path:relative_path>")
def artifacts(relative_path: str):
    full_path = (PROJECT_ROOT / relative_path).resolve()
    if PROJECT_ROOT not in full_path.parents and full_path != PROJECT_ROOT:
        return jsonify({"error": "invalid artifact path"}), 400
    if not full_path.exists():
        return jsonify({"error": "artifact not found"}), 404
    return send_from_directory(full_path.parent, full_path.name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
