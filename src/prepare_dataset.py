from __future__ import annotations

import csv
import struct
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "bodhidharma" / "bodhidharma"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
METADATA_CSV = OUTPUT_DIR / "bodhidharma_metadata.csv"
SUMMARY_CSV = OUTPUT_DIR / "bodhidharma_label_summary.csv"
SUBSET_CSV = OUTPUT_DIR / "bodhidharma_balanced_subset_10.csv"


def read_midi_header(path: Path) -> dict[str, int | str]:
    """Read the standard MIDI header without depending on external packages."""
    with path.open("rb") as file:
        chunk_id = file.read(4)
        if chunk_id != b"MThd":
            return {
                "midi_format": "",
                "track_count": "",
                "ticks_per_beat": "",
                "parse_status": "invalid_header",
            }

        header_length = struct.unpack(">I", file.read(4))[0]
        header = file.read(header_length)
        if len(header) < 6:
            return {
                "midi_format": "",
                "track_count": "",
                "ticks_per_beat": "",
                "parse_status": "short_header",
            }

        midi_format, track_count, division = struct.unpack(">HHH", header[:6])
        ticks_per_beat = division if division < 0x8000 else ""
        return {
            "midi_format": midi_format,
            "track_count": track_count,
            "ticks_per_beat": ticks_per_beat,
            "parse_status": "ok",
        }


def collect_metadata() -> list[dict[str, str | int]]:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    rows: list[dict[str, str | int]] = []
    genre_dirs = sorted(path for path in DATASET_ROOT.iterdir() if path.is_dir())

    for genre_dir in genre_dirs:
        midi_files = sorted(
            path
            for path in genre_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
        )
        for index, midi_path in enumerate(midi_files, start=1):
            header = read_midi_header(midi_path)
            rows.append(
                {
                    "genre": genre_dir.name,
                    "genre_index": index,
                    "file_name": midi_path.name,
                    "relative_path": midi_path.relative_to(PROJECT_ROOT).as_posix(),
                    "size_bytes": midi_path.stat().st_size,
                    **header,
                }
            )

    return rows


def write_csv(path: Path, rows: list[dict[str, str | int]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, str | int]]) -> list[dict[str, str | int]]:
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        genre = str(row["genre"])
        entry = summary.setdefault(
            genre,
            {"file_count": 0, "valid_midi_count": 0, "total_size_bytes": 0},
        )
        entry["file_count"] += 1
        entry["total_size_bytes"] += int(row["size_bytes"])
        if row["parse_status"] == "ok":
            entry["valid_midi_count"] += 1

    return [
        {
            "genre": genre,
            "file_count": values["file_count"],
            "valid_midi_count": values["valid_midi_count"],
            "total_size_bytes": values["total_size_bytes"],
        }
        for genre, values in sorted(summary.items())
    ]


def build_balanced_subset(
    rows: list[dict[str, str | int]], samples_per_genre: int = 10
) -> list[dict[str, str | int]]:
    subset: list[dict[str, str | int]] = []
    by_genre: dict[str, list[dict[str, str | int]]] = {}
    for row in rows:
        if row["parse_status"] == "ok":
            by_genre.setdefault(str(row["genre"]), []).append(row)

    for genre in sorted(by_genre):
        subset.extend(by_genre[genre][:samples_per_genre])

    return subset


def main() -> None:
    metadata = collect_metadata()
    fieldnames = [
        "genre",
        "genre_index",
        "file_name",
        "relative_path",
        "size_bytes",
        "midi_format",
        "track_count",
        "ticks_per_beat",
        "parse_status",
    ]
    write_csv(METADATA_CSV, metadata, fieldnames)

    summary = build_summary(metadata)
    write_csv(
        SUMMARY_CSV,
        summary,
        ["genre", "file_count", "valid_midi_count", "total_size_bytes"],
    )

    subset = build_balanced_subset(metadata, samples_per_genre=10)
    write_csv(SUBSET_CSV, subset, fieldnames)

    print(f"metadata: {METADATA_CSV}")
    print(f"summary: {SUMMARY_CSV}")
    print(f"balanced subset: {SUBSET_CSV}")
    print(f"midi files: {len(metadata)}")
    print(f"genres: {len(summary)}")


if __name__ == "__main__":
    main()
