from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBSET_SIZE = 25
MAX_POINTS_PER_SONG = 300
POINTS_CSV = PROJECT_ROOT / "data" / "processed" / f"bodhidharma_subset_{SUBSET_SIZE}_melody_points.csv"
SAMPLED_CSV = PROJECT_ROOT / "data" / "processed" / f"bodhidharma_subset_{SUBSET_SIZE}_melody_points_sampled_{MAX_POINTS_PER_SONG}.csv"
SUMMARY_CSV = PROJECT_ROOT / "data" / "processed" / f"bodhidharma_subset_{SUBSET_SIZE}_curve_summary.csv"


def evenly_sample(group: pd.DataFrame, max_points: int = 300) -> pd.DataFrame:
    group = group.sort_values("point_index").reset_index(drop=True)
    if len(group) <= max_points:
        sampled = group.copy()
    else:
        indices = np.linspace(0, len(group) - 1, max_points).round().astype(int)
        sampled = group.iloc[indices].copy()

    sampled["sample_index"] = np.arange(1, len(sampled) + 1)
    sampled["sample_count"] = len(sampled)
    return sampled


def main() -> None:
    points = pd.read_csv(POINTS_CSV)
    keys = ["song_id", "genre", "file_name", "relative_path"]

    summary = (
        points.groupby(keys)
        .agg(
            point_count=("point_index", "count"),
            pitch_min=("pitch", "min"),
            pitch_max=("pitch", "max"),
            velocity_mean=("velocity", "mean"),
            duration_beat_mean=("duration_beat", "mean"),
            time_norm_min=("time_norm", "min"),
            time_norm_max=("time_norm", "max"),
        )
        .reset_index()
    )
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")

    sampled_groups = []
    for _, group in points.groupby(keys, sort=False):
        sampled_groups.append(evenly_sample(group, max_points=MAX_POINTS_PER_SONG))
    sampled = pd.concat(sampled_groups, ignore_index=True)
    sampled.to_csv(SAMPLED_CSV, index=False, encoding="utf-8")

    print(f"summary: {SUMMARY_CSV}")
    print(f"sampled points: {SAMPLED_CSV}")
    print(f"songs: {summary.shape[0]}")
    print(f"original points: {points.shape[0]}")
    print(f"sampled points: {sampled.shape[0]}")
    print(summary["point_count"].describe().to_string())


if __name__ == "__main__":
    main()
