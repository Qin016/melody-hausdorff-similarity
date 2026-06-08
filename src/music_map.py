from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.manifold import MDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
HTML_DIR = FIGURES_DIR / "interactive"

POINTS_CSV = PROCESSED_DIR / "bodhidharma_subset_25_melody_points_sampled_300.csv"
DISTANCE_CSV = PROCESSED_DIR / "hausdorff_subset_25_distance_matrix.csv"
MAP_CSV = PROCESSED_DIR / "music_map_mds.csv"
MAP_PNG = FIGURES_DIR / "music_map_mds.png"
MAP_HTML = HTML_DIR / "music_map_mds_interactive.html"


def load_song_metadata() -> pd.DataFrame:
    points = pd.read_csv(POINTS_CSV)
    return (
        points[["song_id", "genre", "file_name", "relative_path"]]
        .drop_duplicates()
        .sort_values(["genre", "file_name"])
        .reset_index(drop=True)
    )


def main() -> None:
    songs = load_song_metadata()
    distances = pd.read_csv(DISTANCE_CSV, index_col=0)

    model = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
        normalized_stress="auto",
        n_init=4,
        max_iter=300,
    )
    coordinates = model.fit_transform(distances.to_numpy())

    music_map = songs.copy()
    music_map["x"] = coordinates[:, 0]
    music_map["y"] = coordinates[:, 1]
    music_map["label"] = music_map["genre"] + " | " + music_map["file_name"]
    music_map.to_csv(MAP_CSV, index=False, encoding="utf-8")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 9), facecolor="#f8fafc")
    ax = plt.gca()
    for genre, group in music_map.groupby("genre"):
        ax.scatter(group["x"], group["y"], s=36, alpha=0.78, label=genre)
    ax.set_title("Music Map Based on Hausdorff Melody-Curve Distance")
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(MAP_PNG, dpi=220)
    plt.close()

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    fig = px.scatter(
        music_map,
        x="x",
        y="y",
        color="genre",
        hover_data=["song_id", "file_name", "relative_path"],
        title="Interactive Music Map: Hausdorff Melody-Curve Distance",
        width=1100,
        height=780,
    )
    fig.update_traces(marker=dict(size=8, opacity=0.82, line=dict(width=0.5, color="#1f2937")))
    fig.update_layout(
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#ffffff",
        legend_title_text="Genre",
        xaxis_title="MDS dimension 1",
        yaxis_title="MDS dimension 2",
    )
    fig.write_html(MAP_HTML, include_plotlyjs=True, full_html=True)

    print(f"songs: {len(music_map)}")
    print(f"stress: {model.stress_:.4f}")
    print(f"map csv: {MAP_CSV}")
    print(f"map png: {MAP_PNG}")
    print(f"interactive map: {MAP_HTML}")


if __name__ == "__main__":
    main()
