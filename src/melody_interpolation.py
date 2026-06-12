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
TEMPO_BPM = 140
MAJOR_SCALE = np.array([0, 2, 4, 5, 7, 9, 11])
CHORD_PATTERN = (0, 5, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在两首歌曲的三维旋律曲线之间插值，生成新旋律。")
    parser.add_argument("--song-a", type=int, help="第一首歌的 song_id。默认自动选择相似曲对。")
    parser.add_argument("--song-b", type=int, help="第二首歌的 song_id。默认自动选择相似曲对。")
    parser.add_argument("--alpha", type=float, default=0.5, help="插值系数，1.0 接近 A，0.0 接近 B。")
    parser.add_argument("--min-distance", type=float, default=0.08, help="自动选择曲对时排除过近重复曲目的最小距离。")
    parser.add_argument("--note-count", type=int, default=180, help="生成 MIDI 使用的音符数量。")
    parser.add_argument("--smooth-window", type=int, default=9, help="旋律曲线平滑窗口，1 表示不平滑。")
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


def smooth_curve(curve: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return curve

    window_size = min(window_size, len(curve))
    if window_size % 2 == 0:
        window_size -= 1
    if window_size <= 1:
        return curve

    radius = window_size // 2
    x = np.arange(-radius, radius + 1)
    sigma = max(1.0, window_size / 3)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    smoothed = curve.copy()
    for dim in (1, 2):
        padded = np.pad(curve[:, dim], (radius, radius), mode="edge")
        smoothed[:, dim] = np.convolve(padded, kernel, mode="valid")
    smoothed[:, 0] = np.linspace(0, 1, len(curve))
    return np.clip(smoothed, 0, 1)


def quantize_to_scale(pitch: int, root: int = 60) -> int:
    candidates: list[int] = []
    for octave in range(-3, 4):
        candidates.extend(int(root + octave * 12 + degree) for degree in MAJOR_SCALE)
    return min(candidates, key=lambda candidate: abs(candidate - pitch))


def build_note_sequence(curve: np.ndarray, ticks_per_beat: int = 480) -> pd.DataFrame:
    base_pitches = np.rint(55 + curve[:, 1] * 24).astype(int)
    velocities = np.rint(52 + curve[:, 2] * 58).astype(int)
    velocities = np.clip(velocities, 38, 116)

    rows: list[dict[str, float | int | bool]] = []
    start_beat = 0.0
    last_pitch: int | None = None

    for index, (point, raw_pitch, velocity) in enumerate(zip(curve, base_pitches, velocities), start=1):
        phrase_pos = (index - 1) % 16
        next_pitch = base_pitches[min(index, len(base_pitches) - 1)]
        pitch_motion = abs(int(next_pitch) - int(raw_pitch))

        if phrase_pos in {3, 7, 11, 15}:
            duration_beat = 0.75
        elif pitch_motion >= 5:
            duration_beat = 0.25
        elif pitch_motion <= 1 and phrase_pos not in {0, 8}:
            duration_beat = 0.5
        else:
            duration_beat = 0.375

        rest = phrase_pos == 15 and index < len(curve)
        pitch = quantize_to_scale(int(raw_pitch))
        if last_pitch is not None and pitch == last_pitch and phrase_pos not in {3, 7, 11, 15}:
            direction = 1 if raw_pitch >= last_pitch else -1
            pitch = quantize_to_scale(pitch + direction * 2)
        pitch = int(np.clip(pitch, 48, 84))

        phrase_gain = 0.78 + 0.22 * np.sin(np.pi * phrase_pos / 15)
        midi_velocity = int(np.clip(velocity * phrase_gain, 35, 120))

        rows.append(
            {
                "time_norm": float(point[0]),
                "pitch_norm": float(point[1]),
                "velocity_norm": float(point[2]),
                "point_index": index,
                "midi_pitch": pitch,
                "midi_velocity": midi_velocity,
                "start_beat": round(start_beat, 3),
                "duration_beat": duration_beat,
                "duration_tick": int(round(duration_beat * ticks_per_beat)),
                "rest": rest,
                "phrase_index": (index - 1) // 16 + 1,
            }
        )
        start_beat += duration_beat
        if not rest:
            last_pitch = pitch

    return pd.DataFrame(rows)


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


def tempo_event(bpm: int) -> bytes:
    microseconds_per_beat = int(round(60_000_000 / bpm))
    return b"\xFF\x51\x03" + microseconds_per_beat.to_bytes(3, "big")


def note_name_to_midi(root: int, degree: int, octave_offset: int = 0) -> int:
    return int(root + octave_offset * 12 + MAJOR_SCALE[degree % len(MAJOR_SCALE)])


def build_melody_track(notes: pd.DataFrame, ticks_per_beat: int) -> bytes:
    track = bytearray()
    track += midi_track_event(0, b"\xFF\x03" + bytes([len(b"Interpolated Melody")]) + b"Interpolated Melody")
    track += midi_track_event(0, tempo_event(TEMPO_BPM))
    track += midi_track_event(0, b"\xC0\x18")  # nylon-string acoustic guitar

    pending_ticks = 0
    for row in notes.itertuples(index=False):
        duration_tick = int(row.duration_tick)
        if bool(row.rest):
            pending_ticks += duration_tick
            continue
        pitch = int(row.midi_pitch)
        velocity = int(row.midi_velocity)
        sounding_ticks = max(1, int(duration_tick * 0.98))
        gap_ticks = max(0, duration_tick - sounding_ticks)
        track += midi_track_event(pending_ticks, bytes([0x90, pitch, velocity]))
        track += midi_track_event(sounding_ticks, bytes([0x80, pitch, 0]))
        pending_ticks = gap_ticks

    track += midi_track_event(pending_ticks, b"\xFF\x2F\x00")
    return bytes(track)


def build_harmony_track(notes: pd.DataFrame, ticks_per_beat: int) -> bytes:
    total_beats = float((notes["start_beat"] + notes["duration_beat"]).max())
    phrase_beats = 4.0
    root = 48
    track = bytearray()
    track += midi_track_event(0, b"\xFF\x03" + bytes([len(b"Harmony Pad")]) + b"Harmony Pad")
    track += midi_track_event(0, b"\xC1\x19")  # steel-string acoustic guitar

    pending_ticks = 0
    phrase_count = int(np.ceil(total_beats / phrase_beats))
    for phrase_index in range(phrase_count):
        degree = CHORD_PATTERN[phrase_index % len(CHORD_PATTERN)]
        chord = [
            note_name_to_midi(root, degree, 0),
            note_name_to_midi(root, degree + 2, 0),
            note_name_to_midi(root, degree + 4, 0),
        ]
        chord_ticks = int(phrase_beats * ticks_per_beat)
        for note_index, pitch in enumerate(chord):
            track += midi_track_event(pending_ticks if note_index == 0 else 0, bytes([0x91, pitch, 42]))
            pending_ticks = 0
        for note_index, pitch in enumerate(chord):
            track += midi_track_event(chord_ticks if note_index == 0 else 0, bytes([0x81, pitch, 0]))

    track += midi_track_event(0, b"\xFF\x2F\x00")
    return bytes(track)


def notes_to_midi(notes: pd.DataFrame, path: Path, ticks_per_beat: int = 480) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    melody_track = build_melody_track(notes, ticks_per_beat)
    harmony_track = build_harmony_track(notes, ticks_per_beat)
    header = b"MThd" + struct.pack(">IHHH", 6, 1, 2, ticks_per_beat)
    chunks = [
        b"MTrk" + struct.pack(">I", len(melody_track)) + melody_track,
        b"MTrk" + struct.pack(">I", len(harmony_track)) + harmony_track,
    ]
    path.write_bytes(header + b"".join(chunks))


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
    if args.note_count < 2:
        raise ValueError("--note-count 至少为 2。")
    if args.smooth_window < 1:
        raise ValueError("--smooth-window 至少为 1。")

    points = load_points()
    if args.song_a is None or args.song_b is None:
        song_a, song_b = choose_pair(args.min_distance)
    else:
        song_a, song_b = args.song_a, args.song_b

    meta_a, curve_a = get_curve(points, song_a)
    meta_b, curve_b = get_curve(points, song_b)
    mixed = interpolate(curve_a, curve_b, alpha=args.alpha, note_count=args.note_count)
    mixed = smooth_curve(mixed, args.smooth_window)

    output = build_note_sequence(mixed)
    output["source_song_a"] = song_a
    output["source_song_b"] = song_b
    output["alpha"] = args.alpha
    output["smooth_window"] = args.smooth_window
    output.to_csv(INTERPOLATED_POINTS_CSV, index=False, encoding="utf-8")

    notes_to_midi(output, GENERATED_MIDI)
    label_a = f"A: {meta_a['genre']} / {meta_a['file_name']}"
    label_b = f"B: {meta_b['genre']} / {meta_b['file_name']}"
    plot_interpolation(resample_curve(curve_a, args.note_count), resample_curve(curve_b, args.note_count), mixed, label_a, label_b)

    print(f"song A: {song_a} | {meta_a['genre']} | {meta_a['file_name']}")
    print(f"song B: {song_b} | {meta_b['genre']} | {meta_b['file_name']}")
    print(f"alpha: {args.alpha}")
    print(f"smooth window: {args.smooth_window}")
    print(f"interpolated points: {INTERPOLATED_POINTS_CSV}")
    print(f"generated midi: {GENERATED_MIDI}")
    print(f"figure: {INTERPOLATION_PNG}")


if __name__ == "__main__":
    main()
