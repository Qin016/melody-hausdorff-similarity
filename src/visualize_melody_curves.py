from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POINTS_CSV = PROJECT_ROOT / "data" / "processed" / "bodhidharma_melody_points_sampled_300.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
HTML_DIR = FIGURES_DIR / "interactive"

SINGLE_PNG = FIGURES_DIR / "melody_curve_3d_single_neon.png"
COMPARISON_PNG = FIGURES_DIR / "melody_curve_3d_genre_comparison.png"
INTERACTIVE_HTML = HTML_DIR / "melody_curves_3d_interactive.html"


GENRE_COLORS = {
    "Country": "#f4c542",
    "Jazz": "#3ddc97",
    "Modern Pop": "#ff5c8a",
    "Rap": "#a56eff",
    "Rhythm and Blues": "#00b4d8",
    "Rock": "#ff6b35",
    "Western Classical": "#4cc9f0",
    "Western Folk": "#80ed99",
    "Worldbeat": "#f72585",
}


def load_points() -> pd.DataFrame:
    points = pd.read_csv(POINTS_CSV)
    return points.sort_values(["song_id", "sample_index"]).reset_index(drop=True)


def choose_representatives(points: pd.DataFrame) -> pd.DataFrame:
    song_counts = (
        points.groupby(["song_id", "genre", "file_name"])
        .size()
        .reset_index(name="point_count")
        .sort_values(["genre", "point_count"], ascending=[True, False])
    )
    return song_counts.groupby("genre", as_index=False).head(1)


def set_dark_3d_style(ax) -> None:
    ax.set_facecolor("#070a12")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor("#101522")
        axis.pane.set_edgecolor("#2a3040")
        axis._axinfo["grid"]["color"] = "#293241"
        axis._axinfo["tick"]["color"] = "#d8dee9"
    ax.tick_params(colors="#d8dee9", labelsize=8)
    ax.xaxis.label.set_color("#d8dee9")
    ax.yaxis.label.set_color("#d8dee9")
    ax.zaxis.label.set_color("#d8dee9")
    ax.title.set_color("#f8f9fa")


def plot_single_neon(points: pd.DataFrame, song_id: int | None = None) -> None:
    if song_id is None:
        representatives = choose_representatives(points)
        song_id = int(representatives.loc[representatives["genre"] == "Jazz", "song_id"].iloc[0])

    curve = points[points["song_id"] == song_id].copy()
    title = f"{curve['genre'].iloc[0]} - {curve['file_name'].iloc[0]}"
    xyz = curve[["time_norm", "pitch_norm", "velocity_norm"]].to_numpy(float)
    segments = np.stack([xyz[:-1], xyz[1:]], axis=1)
    colors = plt.cm.plasma(np.linspace(0, 1, len(segments)))

    fig = plt.figure(figsize=(12, 8), facecolor="#070a12")
    ax = fig.add_subplot(111, projection="3d")
    set_dark_3d_style(ax)

    glow_widths = [9, 6, 3]
    glow_alphas = [0.08, 0.12, 0.20]
    for width, alpha in zip(glow_widths, glow_alphas):
        glow = Line3DCollection(segments, colors=colors, linewidths=width, alpha=alpha)
        ax.add_collection3d(glow)

    line = Line3DCollection(segments, colors=colors, linewidths=2.2, alpha=0.95)
    ax.add_collection3d(line)
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=np.linspace(0, 1, len(xyz)),
        cmap="plasma",
        s=14,
        alpha=0.78,
        depthshade=False,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Pitch")
    ax.set_zlabel("Velocity")
    ax.set_title(f"3D Melody Curve\n{title}", pad=20, fontsize=15)
    ax.view_init(elev=25, azim=235)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(SINGLE_PNG, dpi=240, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_genre_comparison(points: pd.DataFrame) -> None:
    representatives = choose_representatives(points)
    fig = plt.figure(figsize=(14, 9), facecolor="#070a12")
    ax = fig.add_subplot(111, projection="3d")
    set_dark_3d_style(ax)

    for _, row in representatives.iterrows():
        curve = points[points["song_id"] == int(row["song_id"])]
        xyz = curve[["time_norm", "pitch_norm", "velocity_norm"]].to_numpy(float)
        color = GENRE_COLORS.get(row["genre"], "#ffffff")
        ax.plot(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            color=color,
            linewidth=2.1,
            alpha=0.88,
            label=row["genre"],
        )
        ax.scatter(xyz[::12, 0], xyz[::12, 1], xyz[::12, 2], color=color, s=9, alpha=0.6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Pitch")
    ax.set_zlabel("Velocity")
    ax.set_title("Representative 3D Melody Curves by Genre", pad=20, fontsize=15)
    ax.view_init(elev=24, azim=232)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize=8)
    legend.get_frame().set_facecolor("#101522")
    legend.get_frame().set_edgecolor("#2a3040")
    for text in legend.get_texts():
        text.set_color("#f8f9fa")
    plt.tight_layout()
    plt.savefig(COMPARISON_PNG, dpi=240, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_interactive(points: pd.DataFrame) -> None:
    representatives = choose_representatives(points)
    fig = go.Figure()

    for _, row in representatives.iterrows():
        curve = points[points["song_id"] == int(row["song_id"])]
        color = GENRE_COLORS.get(row["genre"], "#ffffff")
        hover = (
            "Genre: "
            + curve["genre"].astype(str)
            + "<br>Song: "
            + curve["file_name"].astype(str)
            + "<br>Point: "
            + curve["sample_index"].astype(str)
            + "<br>Time: "
            + curve["time_norm"].round(3).astype(str)
            + "<br>Pitch: "
            + curve["pitch_norm"].round(3).astype(str)
            + "<br>Velocity: "
            + curve["velocity_norm"].round(3).astype(str)
        )
        fig.add_trace(
            go.Scatter3d(
                x=curve["time_norm"],
                y=curve["pitch_norm"],
                z=curve["velocity_norm"],
                mode="lines+markers",
                name=str(row["genre"]),
                line=dict(color=color, width=5),
                marker=dict(size=3, color=curve["sample_index"], colorscale="Turbo", opacity=0.75),
                text=hover,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="Interactive 3D Melody Curves: Time x Pitch x Velocity",
        paper_bgcolor="#070a12",
        plot_bgcolor="#070a12",
        font=dict(color="#f8f9fa"),
        scene=dict(
            bgcolor="#070a12",
            xaxis=dict(title="Time", gridcolor="#293241", zerolinecolor="#3a4456"),
            yaxis=dict(title="Pitch", gridcolor="#293241", zerolinecolor="#3a4456"),
            zaxis=dict(title="Velocity", gridcolor="#293241", zerolinecolor="#3a4456"),
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
        ),
        legend=dict(bgcolor="rgba(16,21,34,0.85)", bordercolor="#2a3040", borderwidth=1),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    HTML_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(INTERACTIVE_HTML, include_plotlyjs=True, full_html=True)


def main() -> None:
    points = load_points()
    plot_single_neon(points)
    plot_genre_comparison(points)
    make_interactive(points)
    print(f"single curve: {SINGLE_PNG}")
    print(f"genre comparison: {COMPARISON_PNG}")
    print(f"interactive html: {INTERACTIVE_HTML}")


if __name__ == "__main__":
    main()
