from __future__ import annotations

import csv
import struct
from collections import defaultdict, deque
from pathlib import Path
from typing import BinaryIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBSET_CSV = PROJECT_ROOT / "data" / "processed" / "bodhidharma_balanced_subset_10.csv"
NOTES_CSV = PROJECT_ROOT / "data" / "processed" / "bodhidharma_subset_notes.csv"
POINTS_CSV = PROJECT_ROOT / "data" / "processed" / "bodhidharma_subset_melody_points.csv"


def read_varlen(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, offset


def read_chunk(file: BinaryIO) -> tuple[bytes, bytes] | None:
    chunk_id = file.read(4)
    if not chunk_id:
        return None
    if len(chunk_id) != 4:
        raise ValueError("Unexpected end of file while reading chunk id")
    length = struct.unpack(">I", file.read(4))[0]
    return chunk_id, file.read(length)


def parse_midi(path: Path) -> tuple[int, list[dict[str, int]]]:
    with path.open("rb") as file:
        header = read_chunk(file)
        if header is None or header[0] != b"MThd":
            raise ValueError(f"Invalid MIDI header: {path}")

        header_data = header[1]
        if len(header_data) < 6:
            raise ValueError(f"Short MIDI header: {path}")

        _, track_count, division = struct.unpack(">HHH", header_data[:6])
        if division >= 0x8000:
            raise ValueError(f"SMPTE timing is not supported: {path}")
        ticks_per_beat = division

        notes: list[dict[str, int]] = []
        for track_index in range(track_count):
            chunk = read_chunk(file)
            if chunk is None:
                break
            chunk_id, track_data = chunk
            if chunk_id != b"MTrk":
                continue
            notes.extend(parse_track(track_data, track_index))

    return ticks_per_beat, notes


def parse_track(track_data: bytes, track_index: int) -> list[dict[str, int]]:
    offset = 0
    abs_tick = 0
    running_status: int | None = None
    active: dict[tuple[int, int], deque[tuple[int, int]]] = defaultdict(deque)
    notes: list[dict[str, int]] = []

    while offset < len(track_data):
        delta, offset = read_varlen(track_data, offset)
        abs_tick += delta

        status = track_data[offset]
        if status & 0x80:
            offset += 1
            running_status = status
        elif running_status is not None:
            status = running_status
        else:
            raise ValueError("Running status used before status byte")

        if status == 0xFF:
            meta_type = track_data[offset]
            offset += 1
            length, offset = read_varlen(track_data, offset)
            offset += length
            if meta_type == 0x2F:
                break
            continue

        if status in {0xF0, 0xF7}:
            length, offset = read_varlen(track_data, offset)
            offset += length
            continue

        event_type = status & 0xF0
        channel = status & 0x0F
        data_len = 1 if event_type in {0xC0, 0xD0} else 2
        event_data = track_data[offset : offset + data_len]
        offset += data_len

        if event_type not in {0x80, 0x90}:
            continue

        pitch = event_data[0]
        velocity = event_data[1]
        key = (channel, pitch)

        if event_type == 0x90 and velocity > 0:
            active[key].append((abs_tick, velocity))
            continue

        if active[key]:
            onset_tick, onset_velocity = active[key].popleft()
            duration_tick = max(0, abs_tick - onset_tick)
            notes.append(
                {
                    "track": track_index,
                    "channel": channel,
                    "pitch": pitch,
                    "velocity": onset_velocity,
                    "onset_tick": onset_tick,
                    "duration_tick": duration_tick,
                    "end_tick": abs_tick,
                }
            )

    return notes


def read_subset() -> list[dict[str, str]]:
    with SUBSET_CSV.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_melody_points(notes: list[dict[str, object]]) -> list[dict[str, object]]:
    # Channel 9 is percussion in General MIDI. It is excluded from melody curves.
    pitched = [note for note in notes if int(note["channel"]) != 9 and int(note["duration_tick"]) > 0]
    if not pitched:
        return []

    by_onset: dict[int, list[dict[str, object]]] = defaultdict(list)
    for note in pitched:
        by_onset[int(note["onset_tick"])].append(note)

    selected: list[dict[str, object]] = []
    for onset in sorted(by_onset):
        candidates = by_onset[onset]
        selected.append(
            max(
                candidates,
                key=lambda note: (
                    int(note["pitch"]),
                    int(note["velocity"]),
                    int(note["duration_tick"]),
                ),
            )
        )

    max_tick = max(int(note["onset_tick"]) for note in selected)
    min_pitch = min(int(note["pitch"]) for note in selected)
    max_pitch = max(int(note["pitch"]) for note in selected)
    pitch_span = max(1, max_pitch - min_pitch)
    time_span = max(1, max_tick)

    points: list[dict[str, object]] = []
    for point_index, note in enumerate(selected, start=1):
        onset_tick = int(note["onset_tick"])
        pitch = int(note["pitch"])
        velocity = int(note["velocity"])
        points.append(
            {
                **note,
                "point_index": point_index,
                "time_norm": onset_tick / time_span,
                "pitch_norm": (pitch - min_pitch) / pitch_span,
                "velocity_norm": velocity / 127,
            }
        )

    return points


def main() -> None:
    subset = read_subset()
    all_notes: list[dict[str, object]] = []
    all_points: list[dict[str, object]] = []
    failures: list[str] = []

    for song_id, row in enumerate(subset, start=1):
        midi_path = PROJECT_ROOT / row["relative_path"]
        try:
            ticks_per_beat, notes = parse_midi(midi_path)
        except Exception as exc:
            failures.append(f"{row['relative_path']}: {exc}")
            continue

        song_notes: list[dict[str, object]] = []
        for note_index, note in enumerate(sorted(notes, key=lambda n: (n["onset_tick"], n["pitch"])), start=1):
            enriched = {
                "song_id": song_id,
                "genre": row["genre"],
                "file_name": row["file_name"],
                "relative_path": row["relative_path"],
                "note_index": note_index,
                "ticks_per_beat": ticks_per_beat,
                "onset_beat": note["onset_tick"] / ticks_per_beat,
                "duration_beat": note["duration_tick"] / ticks_per_beat,
                **note,
            }
            song_notes.append(enriched)

        points = build_melody_points(song_notes)
        all_notes.extend(song_notes)
        all_points.extend(points)

    note_fields = [
        "song_id",
        "genre",
        "file_name",
        "relative_path",
        "note_index",
        "ticks_per_beat",
        "track",
        "channel",
        "pitch",
        "velocity",
        "onset_tick",
        "duration_tick",
        "end_tick",
        "onset_beat",
        "duration_beat",
    ]
    point_fields = note_fields + ["point_index", "time_norm", "pitch_norm", "velocity_norm"]

    write_csv(NOTES_CSV, all_notes, note_fields)
    write_csv(POINTS_CSV, all_points, point_fields)

    print(f"notes: {NOTES_CSV}")
    print(f"melody points: {POINTS_CSV}")
    print(f"songs processed: {len(subset) - len(failures)}")
    print(f"notes extracted: {len(all_notes)}")
    print(f"melody points: {len(all_points)}")
    if failures:
        print("failures:")
        for failure in failures:
            print(failure)


if __name__ == "__main__":
    main()
