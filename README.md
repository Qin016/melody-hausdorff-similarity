# 基于三维旋律线 Hausdorff 距离的音乐曲风相似性分析

## 1. 研究背景

音乐旋律可以看作随时间变化的音高与音量序列。若将每个音符抽象为三维空间中的一个点，其中三个维度分别表示时间、音高和音量，那么一段旋律就可以表示为由离散音符点连接而成的三维空间曲线。

本项目研究如何利用 Hausdorff 距离度量不同旋律线之间的几何相似性，并进一步分析这种几何度量方法在区分不同音乐曲风方面的有效性。

## 2. 研究目标

本大作业的核心目标包括：

1. 建立音乐旋律线的三维几何表示模型。
2. 从音乐数据中提取时间、音高和音量信息。
3. 使用 Hausdorff 距离计算不同旋律线之间的几何相似性。
4. 通过同曲风与不同曲风样本的对比实验，分析该距离度量在曲风区分中的表现。
5. 总结 Hausdorff 距离用于音乐相似性分析的优势、局限与改进方向。

## 3. 技术路线

```text
音乐片段
  ↓
提取音符信息：时间 t、音高 p、音量 v
  ↓
构造三维点序列 P_i = (t_i, p_i, v_i)
  ↓
归一化处理
  ↓
形成三维旋律折线
  ↓
计算任意两首曲子之间的 Hausdorff 距离
  ↓
得到距离矩阵
  ↓
分析同曲风/异曲风距离差异
  ↓
评估其曲风区分能力
```

## 4. 完整计划表

| 阶段 | 主要任务 | 需要考虑的问题 | 输出成果 |
|---|---|---|---|
| 1. 题目理解与背景调研 | 介绍旋律线、几何相似性、Hausdorff 距离的基本概念 | 为什么音乐可以抽象成几何曲线？Hausdorff 距离适合比较什么类型的对象？ | 背景介绍、研究意义 |
| 2. 音乐数据选择 | 选择若干不同曲风的 MIDI 或可提取音符信息的音乐片段 | 数据来源是否可靠？曲风类别是否清楚？每类样本数量是否均衡？ | 数据集说明表 |
| 3. 音符信息提取 | 从音乐中提取时间、音高、音量三个变量 | 使用 MIDI 还是音频？如果是音频，音高识别难度更大；MIDI 更适合本作业 | 音符序列表 |
| 4. 三维点建模 | 将每个音符表示为三维点：P_i = (t_i, p_i, v_i) | 时间用起始时间还是累计节拍？音高用 MIDI pitch 还是频率？音量是否归一化？ | 旋律点云/曲线模型 |
| 5. 数据预处理 | 对时间、音高、音量进行归一化或标准化 | 三个维度量纲不同，若不处理，时间或音高可能主导距离计算 | 标准化后的三维点 |
| 6. 曲线构造 | 将连续音符点连接成三维折线 | 是否考虑音符持续时间？是否只用主旋律？和弦如何处理？ | 三维旋律曲线图 |
| 7. Hausdorff 距离计算 | 计算两条旋律曲线或点集之间的 Hausdorff 距离 | 使用普通 Hausdorff、平均 Hausdorff，还是改进版本？是否对异常点敏感？ | 距离矩阵 |
| 8. 曲风相似性实验 | 比较同曲风与不同曲风之间的距离 | 同曲风距离是否更小？不同曲风距离是否更大？是否存在混淆？ | 实验结果表、热力图 |
| 9. 分类或聚类验证 | 使用距离矩阵进行 KNN 分类或层次聚类 | Hausdorff 距离能否作为曲风识别特征？准确率如何？ | 分类准确率、聚类图 |
| 10. 结果分析 | 分析距离结果与音乐风格之间的关系 | 哪些曲风容易区分？哪些曲风容易混淆？原因是什么？ | 结果讨论 |
| 11. 局限性分析 | 讨论 Hausdorff 距离的不足 | 对节奏平移、旋律变调、局部异常点是否敏感？是否忽略音乐结构？ | 局限性总结 |
| 12. 改进方向 | 提出更稳健的方案 | 是否可以加入 DTW、Frechet 距离、音程序列、节奏特征等？ | 改进方案 |
| 13. 报告整理 | 完成论文式大作业报告 | 结构是否完整？图表是否充分？数学定义是否清晰？ | 最终报告 |

## 5. 数学建模方法

### 5.1 旋律点表示

设一段旋律包含 n 个音符，第 i 个音符可以表示为：

```text
P_i = (t_i, p_i, v_i)
```

其中：

| 变量 | 含义 | 推荐处理 |
|---|---|---|
| t_i | 第 i 个音符的开始时间或节拍位置 | 归一化到 [0, 1] |
| p_i | 音高，可使用 MIDI pitch | 可使用绝对音高或相对音高 |
| v_i | 音量，可使用 MIDI velocity | 归一化到 [0, 1] |

连续音符点连接后，可以得到一条三维折线，用于表示旋律线。

### 5.2 Hausdorff 距离

设两段旋律对应的点集分别为：

```text
A = {a_1, a_2, ..., a_m}
B = {b_1, b_2, ..., b_n}
```

从 A 到 B 的单向 Hausdorff 距离为：

```text
h(A, B) = max_{a in A} min_{b in B} d(a, b)
```

双向 Hausdorff 距离为：

```text
H(A, B) = max(h(A, B), h(B, A))
```

其中 d(a, b) 通常采用欧氏距离：

```text
d(a, b) = sqrt((t_a - t_b)^2 + (p_a - p_b)^2 + (v_a - v_b)^2)
```

Hausdorff 距离越小，表示两条旋律线在几何形态上越相似。

## 6. 实验设计

### 6.1 数据选择

建议优先使用 MIDI 文件，因为 MIDI 文件天然包含音符开始时间、持续时间、音高和力度信息，适合本题的三维旋律建模。

可选择以下曲风作为实验对象：

| 曲风 | 样本数量建议 |
|---|---|
| 古典 | 5-10 首 |
| 流行 | 5-10 首 |
| 爵士 | 5-10 首 |
| 民谣或民族音乐 | 5-10 首 |

### 6.2 对比实验

为了让实验结果更有说服力，建议设计以下对比：

| 实验 | 目的 |
|---|---|
| 同曲风内部距离比较 | 判断同类音乐是否具有更高几何相似性 |
| 不同曲风之间距离比较 | 判断不同曲风是否能被 Hausdorff 距离区分 |
| 绝对音高实验 | 观察原始旋律几何差异 |
| 相对音高实验 | 减少转调对相似性判断的影响 |
| 普通 Hausdorff 与平均 Hausdorff 对比 | 分析异常点对距离结果的影响 |

### 6.3 评估方式

可使用以下方法评估 Hausdorff 距离的有效性：

1. 距离矩阵热力图：观察同曲风样本是否在矩阵中呈现更小距离。
2. 层次聚类图：观察样本是否能够按照曲风自动聚集。
3. 最近邻分类：对未知样本计算其与训练样本的距离，并归为距离最近样本的曲风。
4. 同曲风与异曲风距离箱线图：比较两类距离分布是否存在明显差异。

## 7. 需要重点考虑的问题

### 7.1 数据层面

- MIDI 数据是否包含清晰的主旋律轨道。
- 多声部音乐中如何提取主旋律。
- 不同曲风样本数量是否均衡。
- 片段长度是否接近，是否需要截取固定长度。

### 7.2 建模层面

- 时间维度使用真实时间还是节拍时间。
- 音高使用绝对音高还是相对音高。
- 音量是否参与距离计算，权重如何设定。
- 三个维度是否需要统一归一化。

### 7.3 距离度量层面

- Hausdorff 距离对异常点较敏感。
- 旋律局部差异可能显著影响整体距离。
- 单纯几何距离可能无法捕捉乐句结构、和声、节奏型和音色差异。
- 两段旋律如果长度不同，需要考虑采样密度和时间归一化。

### 7.4 曲风区分层面

- 曲风不只由旋律决定，还与节奏、和声、配器、音色有关。
- 某些曲风之间可能存在旋律形态重叠。
- Hausdorff 距离更适合作为曲风识别的一个特征，而不是唯一依据。

## 8. 如何让大作业更扎实

建议在最终报告中加入以下内容：

| 加强点 | 作用 |
|---|---|
| 使用真实 MIDI 数据 | 提高实验真实性 |
| 自动提取音符特征 | 提高工程完整性 |
| 三维旋律曲线可视化 | 直观展示建模结果 |
| 距离矩阵热力图 | 清楚呈现样本相似性 |
| 聚类分析 | 验证曲风是否自然分组 |
| 最近邻分类实验 | 提供量化评价指标 |
| 绝对音高与相对音高对比 | 分析转调影响 |
| 普通 Hausdorff 与平均 Hausdorff 对比 | 分析异常点敏感性 |
| 与 DTW 或 Frechet 距离对比 | 提升理论深度 |
| 局限性与改进方向讨论 | 提高报告完整度 |

## 9. 报告建议结构

```text
1. 引言
   1.1 研究背景
   1.2 问题描述
   1.3 本文主要工作

2. 理论基础
   2.1 旋律线的数学表示
   2.2 三维空间曲线建模
   2.3 Hausdorff 距离定义

3. 方法设计
   3.1 数据来源
   3.2 音符特征提取
   3.3 坐标归一化
   3.4 距离计算方法
   3.5 曲风区分实验设计

4. 实验结果
   4.1 三维旋律曲线可视化
   4.2 Hausdorff 距离矩阵
   4.3 同曲风与异曲风距离比较
   4.4 聚类或分类结果

5. 分析与讨论
   5.1 Hausdorff 距离的有效性
   5.2 对曲风区分的贡献
   5.3 局限性分析

6. 改进方向
   6.1 引入 DTW 或 Frechet 距离
   6.2 加入节奏、和声、音色特征
   6.3 使用机器学习分类器

7. 结论
```

## 10. 推荐项目结构

```text
.
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── extract_notes.py
│   ├── melody_curve.py
│   ├── hausdorff.py
│   └── experiment.py
├── notebooks/
│   └── analysis.ipynb
├── figures/
└── report/
```

## 11. 预期结论

本研究预期能够证明：Hausdorff 距离可以在一定程度上反映旋律线的几何相似性，并对部分曲风的区分提供参考。但是，曲风由旋律、节奏、和声、配器和音色等多种因素共同决定，因此仅依赖三维旋律线的 Hausdorff 距离存在一定局限。更稳健的方案是将该距离与节奏特征、和声特征、DTW 距离或机器学习分类模型结合使用。

## 12. 当前实验进展

当前仓库已经完成 Bodhidharma MIDI Dataset 的初步处理与 Hausdorff 距离实验。

已完成内容：

1. 检查 Bodhidharma 数据集完整性。
2. 生成数据集元数据和曲风类别统计。
3. 从每个曲风中抽取 25 首，构造 225 首歌曲的平衡实验子集。
4. 从 MIDI 文件中提取音符事件。
5. 构造三维旋律点：`(time_norm, pitch_norm, velocity_norm)`。
6. 对每首歌的旋律点进行等间隔采样，每首最多 300 点。
7. 计算 90 首歌曲两两之间的 Hausdorff 距离矩阵。
8. 生成距离热力图、同/异曲风箱线图和层次聚类图。
9. 使用 1-NN 最近邻方法测试 Hausdorff 距离的曲风区分能力。
10. 生成三维旋律线可视化，将连续音符点连接为三维空间曲线。
11. 扩展新曲目相似度识别、二维音乐地图和三维曲线插值生成任务。

核心结果：

| 指标 | 结果 |
|---|---:|
| 实验歌曲数 | 225 |
| 曲风类别数 | 9 |
| 两两距离数 | 25,200 |
| 同曲风平均 Hausdorff 距离 | 0.4708 |
| 异曲风平均 Hausdorff 距离 | 0.5003 |
| 1-NN 曲风分类准确率 | 26.67% |

详细实验摘要见：

```text
docs/hausdorff_experiment_summary.md
```

一键运行全流程：

```bash
python run_pipeline.py
```

如果已经生成过结果，只想检查流程是否完整，可以跳过已有输出：

```bash
python run_pipeline.py --skip-existing
```

只运行某几个步骤：

```bash
python run_pipeline.py --steps prepare,extract,resample
python run_pipeline.py --steps hausdorff,visualize
python run_pipeline.py --steps search,map,generate
```

查看所有可用步骤：

```bash
python run_pipeline.py --list-steps
```

单步调试命令：

```bash
python src/prepare_dataset.py
python src/extract_melody_points.py
python src/resample_melody_points.py
python src/hausdorff_experiment.py
python src/visualize_melody_curves.py
python src/similarity_search.py --song-id 11 --top-k 8
python src/music_map.py
python src/melody_interpolation.py --alpha 0.5 --note-count 180
```

三维旋律线可视化输出：

```text
figures/melody_curve_3d_single_neon.png
figures/melody_curve_3d_genre_comparison.png
figures/interactive/melody_curves_3d_interactive.html
```

扩展任务输出：

```text
data/processed/similarity_search_results.csv
figures/similarity_search_top_match_3d.png
data/processed/music_map_mds.csv
figures/music_map_mds.png
figures/interactive/music_map_mds_interactive.html
data/processed/interpolated_melody_points.csv
generated/interpolated_melody.mid
figures/melody_interpolation_3d.png
```

## 13. 前后端分离展示平台

项目提供一个轻量级 Web UI，用于直接在网页中查看实验结果并触发下游任务。

后端 API：

```bash
python backend/app.py
```

前端静态页面：

```bash
python -m http.server 5173 -d frontend
```

然后打开：

```text
http://127.0.0.1:5173
```

前端功能：

1. 展示数据集规模、曲风分布和 Hausdorff 实验指标。
2. 查看三维旋律线静态图和可交互 3D HTML。
3. 选择曲库中的曲目并进行 Top-K 相似曲检索。
4. 查看基于 Hausdorff 距离矩阵降维得到的二维音乐地图。
5. 通过曲线插值生成新的 MIDI 旋律并下载。

## 14. 实验复现指南

本节汇总当前仓库中各个实验部分的复现命令。建议先在项目根目录安装依赖：

```bash
pip install -r requirements.txt
```

如果使用 Windows PowerShell，以下命令均在项目根目录执行。

### 14.1 Bodhidharma 数据集下载

项目默认使用 Bodhidharma MIDI Dataset。原始数据下载并解压到 `data/raw/bodhidharma`：

```powershell
New-Item -ItemType Directory -Force -Path data\raw | Out-Null
Invoke-WebRequest -Uri "https://zenodo.org/records/6959362/files/bodhidharma.zip?download=1" -OutFile "data\raw\bodhidharma.zip"
Expand-Archive -Path "data\raw\bodhidharma.zip" -DestinationPath "data\raw\bodhidharma" -Force
```

解压后应存在类似路径：

```text
data/raw/bodhidharma/bodhidharma/Country/*.mid
data/raw/bodhidharma/bodhidharma/Jazz/*.mid
...
```

### 14.2 平衡子集主实验

平衡子集实验每个曲风取 25 首，共 225 首。可以一键运行完整流程：

```bash
python run_pipeline.py
```

如果已经生成过中间结果，可以跳过已有输出：

```bash
python run_pipeline.py --skip-existing
```

也可以分步运行：

```bash
python src/prepare_dataset.py
python src/extract_melody_points.py
python src/resample_melody_points.py
python src/hausdorff_experiment.py
python src/visualize_melody_curves.py
python src/similarity_search.py --song-id 11 --top-k 8
python src/music_map.py
python src/melody_interpolation.py --alpha 0.5 --note-count 180
```

主要输出：

```text
data/processed/bodhidharma_balanced_subset_25.csv
data/processed/bodhidharma_subset_25_melody_points_sampled_300.csv
data/processed/hausdorff_subset_25_pairwise_distances.csv
data/processed/hausdorff_subset_25_1nn_predictions.csv
figures/hausdorff_distance_heatmap.png
figures/hausdorff_same_vs_diff_boxplot.png
figures/melody_curve_3d_genre_comparison.png
```

### 14.3 采样点数消融实验

用于比较每首歌最大采样点数为 `100, 200, 300, 500, 800, 1000` 时，Hausdorff 1-NN 分类效果是否变化：

```bash
python src/sampling_points_ablation.py
```

主要输出：

```text
data/processed/sampling_points_ablation_summary.csv
data/processed/sampling_points_ablation_predictions.csv
figures/sampling_points_ablation_accuracy.png
figures/sampling_points_ablation_distances.png
```

该实验结论：增加采样点可以略微扩大同/异曲风距离差距，但并没有明显提高 1-NN 分类准确率。

### 14.4 距离方法与机器学习分类器对比

该实验比较 Hausdorff、DTW、Discrete Frechet 的 1-NN 分类效果，并用音乐统计特征训练传统机器学习分类器。

默认使用 80 个旋律采样点运行距离方法：

```bash
python src/method_comparison_experiment.py
```

如果只想跑 300 个采样点下的 DTW 和 Frechet：

```bash
python src/method_comparison_experiment.py --sequence-points 300 --skip-ml --methods "DTW,Discrete Frechet"
```

合并 300 点距离结果与机器学习结果：

```bash
python src/merge_300_point_comparison.py
```

主要输出：

```text
data/processed/distance_method_comparison_summary.csv
data/processed/distance_method_comparison_summary_300.csv
data/processed/ml_song_features.csv
data/processed/ml_classifier_comparison_summary.csv
data/processed/method_comparison_summary_300_combined.csv
figures/method_comparison_accuracy.png
figures/method_comparison_accuracy_300_combined.png
```

该实验结论：DTW 通常略优于 Hausdorff，但提升有限；基于音高、节奏、力度、音程、复音等统计特征的机器学习分类器明显更强。

### 14.5 Bodhidharma 全量非平衡实验

该实验不再构建每类 25 首的平衡子集，而是使用 Bodhidharma 全量 946 首 MIDI。由于类别不平衡，结果同时报告 `accuracy` 和 `balanced accuracy`。

```bash
python src/bodhidharma_full_experiment.py --distance-points 100
```

主要输出：

```text
data/processed/bodhidharma_full_enhanced_song_features.csv
data/processed/bodhidharma_full_melody_points_sampled_300.csv
data/processed/bodhidharma_full_ml_classifier_summary.csv
data/processed/bodhidharma_full_distance_method_summary.csv
figures/bodhidharma_full_ml_accuracy.png
figures/bodhidharma_full_distance_accuracy.png
```

当前全量实验中，SVM RBF 的普通准确率约为 71.67%，balanced accuracy 约为 66.31%；距离方法中 DTW 相对最好，但 balanced accuracy 仍明显低于机器学习分类器。

### 14.6 Lakh MIDI + tagtraum 新数据集实验

为了测试更大规模、更多曲风类别的数据集，项目加入了 Lakh MIDI Dataset matched subset 与 tagtraum/MSD 曲风标签。

下载数据：

```powershell
New-Item -ItemType Directory -Force -Path data\raw\lakh | Out-Null
Invoke-WebRequest -Uri "http://www.tagtraum.com/genres/msd_tagtraum_cd1.cls.zip" -OutFile "data\raw\lakh\msd_tagtraum_cd1.cls.zip"
Invoke-WebRequest -Uri "http://www.tagtraum.com/genres/msd_tagtraum_cd2.cls.zip" -OutFile "data\raw\lakh\msd_tagtraum_cd2.cls.zip"
Invoke-WebRequest -Uri "http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz" -OutFile "data\raw\lakh\lmd_matched.tar.gz"
Expand-Archive -Path data\raw\lakh\msd_tagtraum_cd1.cls.zip -DestinationPath data\raw\lakh -Force
Expand-Archive -Path data\raw\lakh\msd_tagtraum_cd2.cls.zip -DestinationPath data\raw\lakh -Force
```

注意：`lmd_matched.tar.gz` 约 1.4GB，下载和扫描需要较长时间。

构建 14 类、每类 70 首的平衡子集：

```bash
python src/prepare_lakh_dataset.py --per-genre 70 --min-genre-count 70 --seed 42
```

运行 Lakh 实验：

```bash
python src/lakh_genre_experiment.py --distance-per-genre 15
```

主要输出：

```text
data/processed_lakh/lakh_tagtraum_label_summary.csv
data/processed_lakh/lakh_tagtraum_balanced_subset.csv
data/processed_lakh/lakh_enhanced_song_features.csv
data/processed_lakh/lakh_melody_points_sampled_300.csv
data/processed_lakh/lakh_ml_classifier_summary.csv
data/processed_lakh/lakh_distance_method_summary.csv
figures/lakh/lakh_ml_accuracy.png
figures/lakh/lakh_distance_accuracy.png
```

该实验用于验证：当数据规模更大、曲风类别更多且标签更真实复杂时，单一旋律距离方法的区分能力进一步下降，而机器学习特征分类器仍明显优于距离 1-NN。

### 14.7 前后端展示复现

启动后端：

```bash
python backend/app.py
```

启动前端：

```bash
python -m http.server 5173 -d frontend
```

浏览器打开：

```text
http://127.0.0.1:5173
```

页面顶部的 `数据视图` 可以选择：

```text
平衡子集
全量数据集
```

切换后会刷新总览指标、曲风分布、实验结果图、曲目列表和 Top-K 相似曲检索。旋律插值生成目前固定使用平衡子集曲目，因为现有插值脚本依赖平衡子集的曲线和距离文件。

### 14.8 推荐复现实验顺序

如果从零开始，推荐按以下顺序执行：

```text
1. 下载 Bodhidharma 数据集
2. python run_pipeline.py
3. python src/sampling_points_ablation.py
4. python src/method_comparison_experiment.py
5. python src/method_comparison_experiment.py --sequence-points 300 --skip-ml --methods "DTW,Discrete Frechet"
6. python src/merge_300_point_comparison.py
7. python src/bodhidharma_full_experiment.py --distance-points 100
8. 可选：下载 Lakh MIDI，并运行 prepare_lakh_dataset.py 与 lakh_genre_experiment.py
9. 启动 backend/app.py 和 frontend 静态页面查看结果
```
### 14.9 快速复现
若想快速查看前后端,依次运行如下命令，再访问
```text
http://127.0.0.1:5173
```
即可
```text
python run_pipeline.py
python src/bodhidharma_full_experiment.py --distance-points 100
python backend/app.py
python -m http.server 5173 -d frontend
```