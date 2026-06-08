from __future__ import annotations

import argparse
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
GENERATED_DIR = PROJECT_ROOT / "generated"

POINTS_CSV = PROCESSED_DIR / "bodhidharma_subset_25_melody_points_sampled_300.csv"
PAIRWISE_CSV = PROCESSED_DIR / "hausdorff_subset_25_pairwise_distances.csv"
INTERPOLATED_POINTS_CSV = PROCESSED_DIR / "interpolated_melody_points.csv"
INTERPOLATION_PNG = FIGURES_DIR / "melody_interpolation_3d.png"
GENERATED_MIDI = GENERATED_DIR / "interpolated_melody.mid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在两首歌曲的三维旋律曲线之间插值，生成新旋律。")
    parser.add_argument("--song-a", type=int, help="第一首歌的 song_id。默认自动选择相似曲对。")
    parser.add_argument("--song-b", type=int, help="第二首歌的 song_id。默认自动选择相似曲对。")
    parser.add_argument("--alpha", type=float, default=0.5, help="插值系数，1.0 接近 A，0.0 接近 B。")
    parser.add_argument("--min-distance", type=float, default=0.08, help="自动选择曲对时排除过近重复曲目的最小距离。")
    parser.add_argument("--note-count", type=int, default=180, help="生成 MIDI 使用的音符数量。")
    return parser.parse_args()


def load_points() -> pd.DataFrame:
    return pd.read_csv(POINTS_CSV).sort_values(["song_id", "sample_index"])


def choose_pair(min_distance: float) -> tuple[int, int]:
    pairwise = pd.read_csv(PAIRWISE_CSV)
    candidates = pairwise[pairwise["hausdorff_distance"] >= min_distance].sort_values("hausdorff_distance")
    if candidates.empty:
        raise ValueError("找不到满足最小距离条件的曲对。")
    row = candidates.iloc[0]
    return int(row["song_id_a"]), int(row["song_id_b"])


def get_curve(points: pd.DataFrame, song_id: int) -> tuple[pd.Series, np.ndarray]:
    song_points = points[points["song_id"] == song_id].sort_values("sample_index")
    if song_points.empty:
        raise ValueError(f"找不到 song_id={song_id}")
    meta = song_points[["song_id", "genre", "file_name", "relative_path"]].iloc[0]
    curve = song_points[["time_norm", "pitch_norm", "velocity_norm"]].to_numpy(float)
    return meta, curve


def resample_curve(curve: np.ndarray, target_count: int) -> np.ndarray:
    if len(curve) == target_count:
        return curve
    old_x = np.linspace(0, 1, len(curve))
    new_x = np.linspace(0, 1, target_count)
    columns = [np.interp(new_x, old_x, curve[:, dim]) for dim in range(curve.shape[1])]
    return np.stack(columns, axis=1)


def interpolate(curve_a: np.ndarray, curve_b: np.ndarray, alpha: float, note_count: int) -> np.ndarray:
    a = resample_curve(curve_a, note_count)
    b = resample_curve(curve_b, note_count)
    mixed = alpha * a + (1 - alpha) * b
    mixed[:, 0] = np.linspace(0, 1, note_count)
    mixed = np.clip(mixed, 0, 1)
    return mixed


def write_varlen(value: int) -> bytes:
    buffer = value & 0x7F
    value >>= 7
    while value:
        buffer <<= 8
        buffer |= ((value & 0x7F) | 0x80)
        value >>= 7
    output = bytearray()
    while True:
        output.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(output)


def midi_track_event(delta: int, payload: bytes) -> bytes:
    return write_varlen(delta) + payload


def curve_to_midi(curve: np.ndarray, path: Path, ticks_per_beat: int = 480) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    note_ticks = ticks_per_beat // 2
    pitches = np.rint(55 + curve[:, 1] * 24).astype(int)
    velocities = np.rint(45 + curve[:, 2] * 70).astype(int)
    pitches = np.clip(pitches, 48, 84)
    velocities = np.clip(velocities, 35, 120)

    track = bytearray()
    track += midi_track_event(0, b"\xFF\x03" + bytes([len(b"Interpolated Melody")]) + b"Interpolated Melody")
    track += midi_track_event(0, b"\xFF\x51\x03\x07\xA1\x20")  # 120 BPM
    track += midi_track_event(0, b"\xC0\x00")

    for pitch, velocity in zip(pitches, velocities):
        track += midi_track_event(0, bytes([0x90, int(pitch), int(velocity)]))
        track += midi_track_event(note_ticks, bytes([0x80, int(pitch), 0]))
    track += midi_track_event(0, b"\xFF\x2F\x00")

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks_per_beat)
    track_chunk = b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    path.write_bytes(header + track_chunk)


def plot_interpolation(curve_a: np.ndarray, curve_b: np.ndarray, mixed: np.ndarray, label_a: str, label_b: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 8), facecolor="#080b12")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#080b12")
    ax.plot(curve_a[:, 0], curve_a[:, 1], curve_a[:, 2], color="#ffcc00", alpha=0.55, linewidth=1.8, label=label_a)
    ax.plot(curve_b[:, 0], curve_b[:, 1], curve_b[:, 2], color="#00e5ff", alpha=0.55, linewidth=1.8, label=label_b)
    ax.plot(mixed[:, 0], mixed[:, 1], mixed[:, 2], color="#ff4dff", linewidth=2.8, label="Interpolated melody")
    ax.scatter(mixed[:, 0], mixed[:, 1], mixed[:, 2], color="#ff4dff", s=10, alpha=0.7)

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
    ax.set_title("Generated Melody by 3D Curve Interpolation", color="#f8f9fa", pad=18)
    legend = ax.legend()
    legend.get_frame().set_facecolor("#101522")
    for text in legend.get_texts():
        text.set_color("#f8f9fa")
    ax.view_init(elev=24, azim=235)
    plt.tight_layout()
    plt.savefig(INTERPOLATION_PNG, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha 必须在 [0, 1] 之间。")

    points = load_points()
    if args.song_a is None or args.song_b is None:
        song_a, song_b = choose_pair(args.min_distance)
    else:
        song_a, song_b = args.song_a, args.song_b

    meta_a, curve_a = get_curve(points, song_a)
    meta_b, curve_b = get_curve(points, song_b)
    mixed = interpolate(curve_a, curve_b, alpha=args.alpha, note_count=args.note_count)

    output = pd.DataFrame(mixed, columns=["time_norm", "pitch_norm", "velocity_norm"])
    output["point_index"] = np.arange(1, len(output) + 1)
    output["source_song_a"] = song_a
    output["source_song_b"] = song_b
    output["alpha"] = args.alpha
    output.to_csv(INTERPOLATED_POINTS_CSV, index=False, encoding="utf-8")

    curve_to_midi(mixed, GENERATED_MIDI)
    label_a = f"A: {meta_a['genre']} / {meta_a['file_name']}"
    label_b = f"B: {meta_b['genre']} / {meta_b['file_name']}"
    plot_interpolation(resample_curve(curve_a, args.note_count), resample_curve(curve_b, args.note_count), mixed, label_a, label_b)

    print(f"song A: {song_a} | {meta_a['genre']} | {meta_a['file_name']}")
    print(f"song B: {song_b} | {meta_b['genre']} | {meta_b['file_name']}")
    print(f"alpha: {args.alpha}")
    print(f"interpolated points: {INTERPOLATED_POINTS_CSV}")
    print(f"generated midi: {GENERATED_MIDI}")
    print(f"figure: {INTERPOLATION_PNG}")


if __name__ == "__main__":
    main()
