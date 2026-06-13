from __future__ import annotations

import argparse
import csv
import random
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "lakh"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_lakh"

LMD_ARCHIVE = RAW_DIR / "lmd_matched.tar.gz"
TAGTRAUM_CD2 = RAW_DIR / "msd_tagtraum_cd2.cls"
SELECTED_MIDI_DIR = RAW_DIR / "selected_midi"
METADATA_CSV = PROCESSED_DIR / "lakh_tagtraum_metadata.csv"
SUBSET_CSV = PROCESSED_DIR / "lakh_tagtraum_balanced_subset.csv"
LABEL_SUMMARY_CSV = PROCESSED_DIR / "lakh_tagtraum_label_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a genre-balanced Lakh MIDI subset with tagtraum labels.")
    parser.add_argument("--per-genre", type=int, default=100, help="Maximum songs selected per genre.")
    parser.add_argument("--min-genre-count", type=int, default=100, help="Minimum matched songs required to keep a genre.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_tagtraum_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cells = line.split("\t")
            if len(cells) >= 2:
                labels[cells[0]] = cells[1].replace("/", "_")
    return labels


def archive_midi_by_track(path: Path) -> dict[str, str]:
    tracks: dict[str, str] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith((".mid", ".midi")):
                continue
            parts = Path(member.name).parts
            track_id = next((part for part in parts if part.startswith("TR") and len(part) >= 18), None)
            if track_id and track_id not in tracks:
                tracks[track_id] = member.name
    return tracks


def extract_selected_midis(archive_path: Path, selected_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    wanted = {row["archive_path"]: row for row in selected_rows}
    extracted_rows: list[dict[str, str]] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            row = wanted.get(member.name)
            if row is None:
                continue
            output_dir = SELECTED_MIDI_DIR / row["genre"]
            output_dir.mkdir(parents=True, exist_ok=True)
            output_name = f"{row['track_id']}__{Path(member.name).name}"
            output_path = output_dir / output_name
            with archive.extractfile(member) as source, output_path.open("wb") as target:
                if source is None:
                    continue
                target.write(source.read())
            row = {**row, "relative_path": output_path.relative_to(PROJECT_ROOT).as_posix()}
            extracted_rows.append(row)
    return extracted_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    labels = read_tagtraum_labels(TAGTRAUM_CD2)
    midi_by_track = archive_midi_by_track(LMD_ARCHIVE)

    rows = []
    for track_id, archive_path in midi_by_track.items():
        genre = labels.get(track_id)
        if genre is None:
            continue
        rows.append({"track_id": track_id, "genre": genre, "archive_path": archive_path})

    rows_by_genre: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_genre[row["genre"]].append(row)

    kept_genres = sorted(genre for genre, items in rows_by_genre.items() if len(items) >= args.min_genre_count)
    selected: list[dict[str, str]] = []
    for genre in kept_genres:
        items = rows_by_genre[genre]
        rng.shuffle(items)
        selected.extend(items[: args.per_genre])

    extracted = extract_selected_midis(LMD_ARCHIVE, selected)
    extracted.sort(key=lambda row: (row["genre"], row["track_id"]))
    for song_id, row in enumerate(extracted, start=1):
        row["song_id"] = song_id
        row["file_name"] = Path(row["relative_path"]).name

    metadata_rows = sorted(rows, key=lambda row: (row["genre"], row["track_id"]))
    write_csv(METADATA_CSV, metadata_rows, ["track_id", "genre", "archive_path"])
    write_csv(SUBSET_CSV, extracted, ["song_id", "track_id", "genre", "file_name", "relative_path", "archive_path"])

    matched_counts = Counter(row["genre"] for row in rows)
    selected_counts = Counter(row["genre"] for row in extracted)
    summary_rows = [
        {
            "genre": genre,
            "matched_count": matched_counts[genre],
            "selected_count": selected_counts[genre],
        }
        for genre in sorted(matched_counts)
    ]
    write_csv(LABEL_SUMMARY_CSV, summary_rows, ["genre", "matched_count", "selected_count"])

    print(f"matched tracks with labels: {len(rows)}")
    print(f"kept genres: {len(kept_genres)}")
    print(f"selected midi files: {len(extracted)}")
    print(f"metadata: {METADATA_CSV}")
    print(f"subset: {SUBSET_CSV}")
    print(f"label summary: {LABEL_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
