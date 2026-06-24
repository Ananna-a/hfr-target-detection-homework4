# 高频地波雷达目标检测实验

本项目用于完成海洋信息技术大作业中的高频地波雷达目标检测方向。程序从课程提供的 `HFRData.mat` 点迹数据出发，完成数据探查、点迹整理、基础可视化、目标候选检测和多帧航迹确认。

## 目录结构

```text
code/
  hfr_config.py           路径、参数、绘图风格和数据读取
  01_prepare_data.py      点迹表格整理
  02_detect_and_track.py  候选检测与多帧航迹确认
  03_make_figures.py      生成报告图件
  04_analyze_results.py   生成实验结果分析

report/
  目标检测原理与工作流程.md
  figures/                运行后生成图片
  tables/                 运行后生成统计表
  results/                运行后生成检测结果
```

## 数据说明

原始数据、学习资料和生成结果不上传到 GitHub。运行前需要在本地保留课程数据：

```text
点迹数据/HFRData.mat
点迹数据/点迹数据字段说明.docx
```

## 环境配置

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 运行流程

```powershell
python code\01_prepare_data.py
python code\02_detect_and_track.py
python code\03_make_figures.py
python code\04_analyze_results.py
```

运行后会生成：

- `report/figures/`：报告图件，同时输出 PNG、SVG、PDF
- `report/tables/`：清洗后的点迹表、数据摘要表和每帧统计表
- `report/results/`：候选目标和确认航迹结果
- `report/实验结果分析.md`：实验结果的文字分析

## 方法概述

由于数据中的 `Class` 字段全部为 4，缺少可直接用于监督学习的目标/杂波标签，因此本项目采用规则与无监督方法结合的流程：

```text
点迹数据 -> 时间字段重建 -> 数据清洗 -> 信噪比和幅度初筛 -> DBSCAN聚类 -> 多帧连续性与形态质量确认
```

该流程不训练神经网络模型，重点是从点迹中提取目标候选，并通过多帧运动连续性、航迹直线性和相邻帧跳变约束降低虚警。报告图件保留点迹空间密度、点迹空间分布、确认航迹和单帧候选检测过程图；多帧航迹确认结果同时采用表格审查，避免把跳变明显的候选链误读为真实航迹。

原始 `raw_unixtime` 存在多帧共用同一值的情况，但这些帧的点迹集合不同，因此项目保留全部帧，并使用帧头年月日时分秒重建逐帧时间。
