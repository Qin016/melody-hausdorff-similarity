from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PipelineStep:
    """描述流水线中的一个步骤。"""

    name: str
    title: str
    script: Path
    outputs: tuple[Path, ...]
    description: str


STEPS: tuple[PipelineStep, ...] = (
    PipelineStep(
        name="prepare",
        title="数据集索引与平衡子集生成",
        script=PROJECT_ROOT / "src" / "prepare_dataset.py",
        outputs=(
            PROJECT_ROOT / "data" / "processed" / "bodhidharma_metadata.csv",
            PROJECT_ROOT / "data" / "processed" / "bodhidharma_label_summary.csv",
            PROJECT_ROOT / "data" / "processed" / "bodhidharma_balanced_subset_10.csv",
        ),
        description="检查原始 MIDI 文件，生成元数据、类别统计和每类 10 首的实验子集。",
    ),
    PipelineStep(
        name="extract",
        title="MIDI 音符事件与三维旋律点提取",
        script=PROJECT_ROOT / "src" / "extract_melody_points.py",
        outputs=(
            PROJECT_ROOT / "data" / "processed" / "bodhidharma_subset_notes.csv",
            PROJECT_ROOT / "data" / "processed" / "bodhidharma_subset_melody_points.csv",
        ),
        description="解析 MIDI 事件，提取 time/pitch/velocity，并连接为三维旋律点序列。",
    ),
    PipelineStep(
        name="resample",
        title="旋律曲线等间隔采样",
        script=PROJECT_ROOT / "src" / "resample_melody_points.py",
        outputs=(
            PROJECT_ROOT / "data" / "processed" / "bodhidharma_curve_summary.csv",
            PROJECT_ROOT / "data" / "processed" / "bodhidharma_melody_points_sampled_300.csv",
        ),
        description="将每首歌的旋律点压缩到最多 300 个点，降低 Hausdorff 距离计算量。",
    ),
    PipelineStep(
        name="hausdorff",
        title="Hausdorff 距离矩阵与曲风区分实验",
        script=PROJECT_ROOT / "src" / "hausdorff_experiment.py",
        outputs=(
            PROJECT_ROOT / "data" / "processed" / "hausdorff_distance_matrix.csv",
            PROJECT_ROOT / "data" / "processed" / "hausdorff_pairwise_distances.csv",
            PROJECT_ROOT / "data" / "processed" / "hausdorff_genre_summary.csv",
            PROJECT_ROOT / "data" / "processed" / "hausdorff_1nn_predictions.csv",
            PROJECT_ROOT / "figures" / "hausdorff_distance_heatmap.png",
            PROJECT_ROOT / "figures" / "hausdorff_same_vs_diff_boxplot.png",
            PROJECT_ROOT / "figures" / "hausdorff_hierarchical_clustering.png",
        ),
        description="计算两两 Hausdorff 距离，生成距离矩阵、统计表、分类结果和可视化图。",
    ),
    PipelineStep(
        name="visualize",
        title="三维旋律线可视化",
        script=PROJECT_ROOT / "src" / "visualize_melody_curves.py",
        outputs=(
            PROJECT_ROOT / "figures" / "melody_curve_3d_single_neon.png",
            PROJECT_ROOT / "figures" / "melody_curve_3d_genre_comparison.png",
            PROJECT_ROOT / "figures" / "interactive" / "melody_curves_3d_interactive.html",
        ),
        description="生成静态 3D 旋律曲线图和可交互 HTML，展示音符点连接形成的空间曲线。",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="一键运行音乐旋律线 Hausdorff 距离大作业实验全流程。"
    )
    parser.add_argument(
        "--steps",
        default="all",
        help=(
            "选择要运行的步骤，默认 all。"
            "可选：prepare,extract,resample,hausdorff,visualize，多个步骤用英文逗号分隔。"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果某一步的输出文件已经存在且非空，则跳过该步骤。",
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="只列出可运行步骤，不真正执行。",
    )
    return parser.parse_args()


def selected_steps(step_text: str) -> list[PipelineStep]:
    if step_text.lower() == "all":
        return list(STEPS)

    requested = [item.strip().lower() for item in step_text.split(",") if item.strip()]
    known = {step.name: step for step in STEPS}
    unknown = [name for name in requested if name not in known]
    if unknown:
        valid = ", ".join(step.name for step in STEPS)
        raise ValueError(f"未知步骤：{', '.join(unknown)}。可选步骤：{valid}")
    return [known[name] for name in requested]


def outputs_ready(step: PipelineStep) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in step.outputs)


def ensure_dataset_exists() -> None:
    dataset_root = PROJECT_ROOT / "data" / "raw" / "bodhidharma" / "bodhidharma"
    if not dataset_root.exists():
        raise FileNotFoundError(
            "未找到 Bodhidharma 数据集目录："
            f"{dataset_root}\n"
            "请先下载并解压 bodhidharma.zip 到 data/raw/bodhidharma/。"
        )


def run_step(step: PipelineStep, skip_existing: bool) -> None:
    print(f"\n=== {step.name}: {step.title} ===")
    print(step.description)

    if skip_existing and outputs_ready(step):
        print("输出文件已存在，跳过该步骤。")
        return

    if not step.script.exists():
        raise FileNotFoundError(f"找不到脚本：{step.script}")

    start = time.perf_counter()
    command = [sys.executable, str(step.script)]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    elapsed = time.perf_counter() - start

    missing = [path for path in step.outputs if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise RuntimeError(f"步骤运行完成，但缺少预期输出：\n{missing_text}")

    print(f"完成：{step.title}，耗时 {elapsed:.1f} 秒")


def print_step_list() -> None:
    print("可运行步骤：")
    for step in STEPS:
        print(f"- {step.name}: {step.title}")
        print(f"  {step.description}")


def main() -> None:
    args = parse_args()

    if args.list_steps:
        print_step_list()
        return

    ensure_dataset_exists()
    steps = selected_steps(args.steps)

    print("音乐旋律线几何相似性实验流水线")
    print(f"项目目录：{PROJECT_ROOT}")
    print(f"运行步骤：{', '.join(step.name for step in steps)}")

    total_start = time.perf_counter()
    for step in steps:
        run_step(step, skip_existing=args.skip_existing)

    total_elapsed = time.perf_counter() - total_start
    print("\n=== 全流程完成 ===")
    print(f"总耗时：{total_elapsed:.1f} 秒")
    print("主要结果位置：")
    print(f"- 实验摘要：{PROJECT_ROOT / 'docs' / 'hausdorff_experiment_summary.md'}")
    print(f"- 距离矩阵：{PROJECT_ROOT / 'data' / 'processed' / 'hausdorff_distance_matrix.csv'}")
    print(f"- 三维可视化 HTML：{PROJECT_ROOT / 'figures' / 'interactive' / 'melody_curves_3d_interactive.html'}")


if __name__ == "__main__":
    main()
