from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extract_melody_points import build_melody_points, parse_midi
from hausdorff_experiment import hausdorff


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
POINTS_CSV = PROCESSED_DIR / "bodhidharma_subset_25_melody_points_sampled_300.csv"
RESULT_CSV = PROCESSED_DIR / "similarity_search_results.csv"
COMPARE_PNG = FIGURES_DIR / "similarity_search_top_match_3d.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于三维旋律线 Hausdorff 距离进行以曲搜曲。")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--song-id", type=int, help="使用曲库中的 song_id 作为查询曲目。")
    input_group.add_argument("--midi", type=Path, help="使用外部 MIDI 文件作为查询曲目。")
    parser.add_argument("--top-k", type=int, default=10, help="返回最相似的 Top-K 曲目。")
    parser.add_argument("--max-points", type=int, default=300, help="外部 MIDI 查询曲线最多采样点数。")
    return parser.parse_args()


def load_library() -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    points = pd.read_csv(POINTS_CSV).sort_values(["song_id", "sample_index"])
    songs = (
        points[["song_id", "genre", "file_name", "relative_path"]]
        .drop_duplicates()
        .sort_values("song_id")
        .reset_index(drop=True)
    )
    curves = {
        int(song_id): group[["time_norm", "pitch_norm", "velocity_norm"]].to_numpy(float)
        for song_id, group in points.groupby("song_id")
    }
    return songs, curves


def evenly_sample_array(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points).round().astype(int)
    return points[indices]


def external_midi_curve(path: Path, max_points: int) -> np.ndarray:
    ticks_per_beat, raw_notes = parse_midi(path)
    song_notes = []
    for note_index, note in enumerate(sorted(raw_notes, key=lambda n: (n["onset_tick"], n["pitch"])), start=1):
        song_notes.append(
            {
                "song_id": 0,
                "genre": "query",
                "file_name": path.name,
                "relative_path": str(path),
                "note_index": note_index,
                "ticks_per_beat": ticks_per_beat,
                "onset_beat": note["onset_tick"] / ticks_per_beat,
                "duration_beat": note["duration_tick"] / ticks_per_beat,
                **note,
            }
        )
    melody_points = build_melody_points(song_notes)
    if not melody_points:
        raise ValueError(f"未能从 MIDI 中提取有效旋律点：{path}")
    curve = np.array(
        [[point["time_norm"], point["pitch_norm"], point["velocity_norm"]] for point in melody_points],
        dtype=float,
    )
    return evenly_sample_array(curve, max_points=max_points)


def query_from_library(song_id: int, curves: dict[int, np.ndarray]) -> np.ndarray:
    if song_id not in curves:
        known = ", ".join(str(item) for item in sorted(curves)[:10])
        raise ValueError(f"找不到 song_id={song_id}。示例可用 song_id：{known} ...")
    return curves[song_id]


def search(
    query_curve: np.ndarray,
    songs: pd.DataFrame,
    curves: dict[int, np.ndarray],
    top_k: int,
    exclude_song_id: int | None = None,
) -> pd.DataFrame:
    rows = []
    for _, song in songs.iterrows():
        song_id = int(song["song_id"])
        if exclude_song_id is not None and song_id == exclude_song_id:
            continue
        distance = hausdorff(query_curve, curves[song_id])
        rows.append(
            {
                "rank": 0,
                "song_id": song_id,
                "genre": song["genre"],
                "file_name": song["file_name"],
                "relative_path": song["relative_path"],
                "hausdorff_distance": distance,
            }
        )
    results = pd.DataFrame(rows).sort_values("hausdorff_distance").head(top_k).reset_index(drop=True)
    results["rank"] = np.arange(1, len(results) + 1)
    return results


def plot_top_match(query_curve: np.ndarray, top_curve: np.ndarray, top_label: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 8), facecolor="#080b12")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#080b12")

    ax.plot(
        query_curve[:, 0],
        query_curve[:, 1],
        query_curve[:, 2],
        color="#ffcc00",
        linewidth=2.4,
        label="Query",
    )
    ax.scatter(query_curve[:, 0], query_curve[:, 1], query_curve[:, 2], color="#ffcc00", s=10, alpha=0.65)
    ax.plot(
        top_curve[:, 0],
        top_curve[:, 1],
        top_curve[:, 2],
        color="#00e5ff",
        linewidth=2.4,
        label=f"Top match: {top_label}",
    )
    ax.scatter(top_curve[:, 0], top_curve[:, 1], top_curve[:, 2], color="#00e5ff", s=10, alpha=0.55)

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
    plt.savefig(COMPARE_PNG, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    songs, curves = load_library()

    if args.song_id is not None:
        query_curve = query_from_library(args.song_id, curves)
        exclude_song_id = args.song_id
        query_label = f"song_id={args.song_id}"
    else:
        query_curve = external_midi_curve(args.midi, max_points=args.max_points)
        exclude_song_id = None
        query_label = str(args.midi)

    results = search(query_curve, songs, curves, top_k=args.top_k, exclude_song_id=exclude_song_id)
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_CSV, index=False, encoding="utf-8")

    top = results.iloc[0]
    plot_top_match(
        query_curve,
        curves[int(top["song_id"])],
        f"{top['genre']} / {top['file_name']}",
    )

    print(f"query: {query_label}")
    print(f"results: {RESULT_CSV}")
    print(f"top-match figure: {COMPARE_PNG}")
    print(results[["rank", "song_id", "genre", "file_name", "hausdorff_distance"]].to_string(index=False))


if __name__ == "__main__":
    main()
