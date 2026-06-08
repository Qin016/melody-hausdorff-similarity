from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
GENERATED_DIR = PROJECT_ROOT / "generated"
SRC_DIR = PROJECT_ROOT / "src"

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


def plot_search_match(query_song_id: int, match_song_id: int) -> None:
    points = read_csv(DATA_DIR / "bodhidharma_subset_25_melody_points_sampled_300.csv")
    query = points[points["song_id"] == query_song_id].sort_values("sample_index")
    match = points[points["song_id"] == match_song_id].sort_values("sample_index")
    if query.empty or match.empty:
        return

    output = FIGURES_DIR / "similarity_search_top_match_3d.png"
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


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "project_root": str(PROJECT_ROOT)})


@app.get("/api/summary")
def summary():
    label_summary = read_csv(DATA_DIR / "bodhidharma_label_summary.csv")
    pairwise = read_csv(DATA_DIR / "hausdorff_subset_25_pairwise_distances.csv")
    knn = read_csv(DATA_DIR / "hausdorff_subset_25_1nn_predictions.csv")
    points = read_csv(DATA_DIR / "bodhidharma_subset_25_melody_points_sampled_300.csv")

    same = pairwise[pairwise["same_genre"]]["hausdorff_distance"]
    diff = pairwise[~pairwise["same_genre"]]["hausdorff_distance"]

    return jsonify(
        {
            "dataset": {
                "midi_files": int(label_summary["file_count"].sum()),
                "genres": int(label_summary.shape[0]),
                "balanced_songs": int(points["song_id"].nunique()),
                "sampled_points": int(points.shape[0]),
                "label_summary": label_summary.to_dict(orient="records"),
            },
            "hausdorff": {
                "pairwise_distances": int(pairwise.shape[0]),
                "same_genre_mean": float(same.mean()),
                "different_genre_mean": float(diff.mean()),
                "same_genre_median": float(same.median()),
                "different_genre_median": float(diff.median()),
                "one_nn_accuracy": float(knn["correct"].mean()),
            },
            "artifacts": {
                "distance_heatmap": artifact_url(FIGURES_DIR / "hausdorff_distance_heatmap.png"),
                "same_diff_boxplot": artifact_url(FIGURES_DIR / "hausdorff_same_vs_diff_boxplot.png"),
                "cluster": artifact_url(FIGURES_DIR / "hausdorff_hierarchical_clustering.png"),
                "single_curve": artifact_url(FIGURES_DIR / "melody_curve_3d_single_neon.png"),
                "genre_curves": artifact_url(FIGURES_DIR / "melody_curve_3d_genre_comparison.png"),
                "interactive_curves": artifact_url(FIGURES_DIR / "interactive" / "melody_curves_3d_interactive.html"),
                "music_map": artifact_url(FIGURES_DIR / "music_map_mds.png"),
                "interactive_music_map": artifact_url(FIGURES_DIR / "interactive" / "music_map_mds_interactive.html"),
                "interpolation": artifact_url(FIGURES_DIR / "melody_interpolation_3d.png"),
                "generated_midi": artifact_url(GENERATED_DIR / "interpolated_melody.mid"),
            },
        }
    )


@app.get("/api/songs")
def songs():
    points = read_csv(DATA_DIR / "bodhidharma_subset_25_melody_points_sampled_300.csv")
    songs_df = (
        points[["song_id", "genre", "file_name", "relative_path"]]
        .drop_duplicates()
        .sort_values(["genre", "file_name"])
        .reset_index(drop=True)
    )
    return jsonify(songs_df.to_dict(orient="records"))


@app.get("/api/search")
def search():
    song_id = request.args.get("song_id", type=int)
    top_k = request.args.get("top_k", default=8, type=int)
    if song_id is None:
        return jsonify({"error": "song_id is required"}), 400

    pairwise = read_csv(DATA_DIR / "hausdorff_subset_25_pairwise_distances.csv")
    songs_df = read_csv(DATA_DIR / "bodhidharma_subset_25_melody_points_sampled_300.csv")
    songs_df = (
        songs_df[["song_id", "genre", "file_name", "relative_path"]]
        .drop_duplicates()
        .set_index("song_id")
    )

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
    results.to_csv(DATA_DIR / "similarity_search_results.csv", index=False, encoding="utf-8")
    plot_search_match(song_id, int(results.iloc[0]["song_id"]))
    return jsonify(
        {
            "query_song_id": song_id,
            "results": results.to_dict(orient="records"),
            "figure": artifact_url(FIGURES_DIR / "similarity_search_top_match_3d.png"),
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
    run_script("melody_interpolation.py", "--alpha", str(alpha), "--note-count", str(note_count))
    return jsonify(
        {
            "alpha": alpha,
            "note_count": note_count,
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
