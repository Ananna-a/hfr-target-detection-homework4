import sys

import pandas as pd

from hfr_config import PROJECT_ROOT, RESULT_DIR, TABLE_DIR


# 分析文档路径来源：报告实验结果章节引用
ANALYSIS_FILE = PROJECT_ROOT / "report" / "实验结果分析.md"
# 展示列来源：报告中保留核心评价指标
DISPLAY_COLUMNS = [
    "track_id",
    "frame_count",
    "start_frame",
    "end_frame",
    "mean_snr",
    "mean_velocity",
    "displacement",
    "path_length",
    "straightness",
    "max_step",
    "max_turn_angle",
]
# 整数列来源：航迹编号和帧编号
INTEGER_COLUMNS = ["track_id", "frame_count", "start_frame", "end_frame"]
# 小数列来源：航迹质量和信号质量
ROUND_COLUMNS = [
    "mean_snr",
    "mean_velocity",
    "displacement",
    "path_length",
    "straightness",
    "max_step",
    "max_turn_angle",
]


def dataframe_to_markdown(data_frame: pd.DataFrame) -> list[str]:
    # 将表格转换为Markdown文本
    header_line = "| " + " | ".join(data_frame.columns) + " |"
    split_line = "| " + " | ".join(["---"] * len(data_frame.columns)) + " |"
    body_lines = []
    for row_data in data_frame.to_dict(orient="records"):
        body_lines.append("| " + " | ".join(str(row_data[column_name]) for column_name in data_frame.columns) + " |")
    return [header_line, split_line, *body_lines]


def get_summary_value(data_summary: pd.DataFrame, metric_name: str) -> str:
    # 读取数据摘要中的指定指标
    return str(data_summary[data_summary["指标"] == metric_name]["数值"].iloc[0])


def build_display_summary(track_summary: pd.DataFrame) -> pd.DataFrame:
    # 整理航迹摘要展示表
    if track_summary.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)
    display_summary = track_summary[DISPLAY_COLUMNS].copy()
    for column_name in INTEGER_COLUMNS:
        display_summary[column_name] = display_summary[column_name].astype(int)
    for column_name in ROUND_COLUMNS:
        display_summary[column_name] = display_summary[column_name].round(2)
    return display_summary


def build_track_review_lines(display_summary: pd.DataFrame) -> list[str]:
    # 生成航迹质量审查文本
    if display_summary.empty:
        return [
            "当前参数下未形成同时满足持续帧数和形态质量约束的疑似目标航迹。",
            "这说明质量过滤后保留结果较保守，可在报告中表述为候选检测演示，不宜宣称完成真实目标确认。",
        ]

    longest_track = display_summary.sort_values(["frame_count", "mean_snr"], ascending=False).iloc[0]
    best_quality_track = display_summary.sort_values(
        ["straightness", "frame_count", "mean_snr"],
        ascending=False,
    ).iloc[0]
    longest_track_id = int(longest_track["track_id"])
    longest_frame_count = int(longest_track["frame_count"])
    best_track_id = int(best_quality_track["track_id"])
    best_frame_count = int(best_quality_track["frame_count"])
    best_straightness = best_quality_track["straightness"]
    best_max_step = best_quality_track["max_step"]
    best_track_text = (
        f"质量较好的候选链为 T{best_track_id}，持续 {best_frame_count} 帧，"
        f"直线性为 {best_straightness:.2f}，最大跳变为 {best_max_step:.2f} km。"
    )
    return [
        *dataframe_to_markdown(display_summary),
        "",
        "表中 `straightness` 为首尾位移与累计路径长度之比，越接近 1 表示轨迹越接近单调运动；`max_step` 为相邻帧中心最大跳变；`max_turn_angle` 为连续两段中心位移的最大夹角。",
        f"最长航迹为 T{longest_track_id}，持续 {longest_frame_count} 帧，直线性为 {longest_track['straightness']:.2f}。",
        best_track_text,
        "由于当前数据没有 AIS 或人工真值，空间折线图只能展示质量确认后的疑似航迹中心移动，不能表述为已验证船舶真实航迹。",
    ]


def build_time_review_lines(frame_summary: pd.DataFrame) -> list[str]:
    # 生成帧时间说明文本
    time_diff = frame_summary["time"].diff().dropna()
    time_diff_values = sorted(int(value) for value in time_diff.unique())
    raw_time_group_count = int((frame_summary.groupby("raw_unixtime").size() > 1).sum())
    raw_time_unique_count = int(frame_summary["raw_unixtime"].nunique())
    frame_time_start = str(frame_summary["frame_time"].iloc[0])
    frame_time_end = str(frame_summary["frame_time"].iloc[-1])
    return [
        "## 帧时间说明",
        "",
        f"原始 `raw_unixtime` 共有 {raw_time_unique_count} 个唯一值，其中 {raw_time_group_count} 组对应多帧。",
        "这些同 `raw_unixtime` 帧的点迹数量、点迹编号和空间分布并不相同，因此不作为重复帧删除。",
        f"本文使用帧头年月日时分秒重建逐帧时间，范围为 {frame_time_start} 至 {frame_time_end}。",
        f"重建后按帧序号相邻比较的时间差为 {time_diff_values} 秒。",
        "",
    ]


def build_metric_review_lines(
    cluster_table: pd.DataFrame,
    track_table: pd.DataFrame,
    confirmed_tracks: pd.DataFrame,
) -> list[str]:
    # 生成无监督评价指标说明
    track_count = track_table["track_id"].nunique() if not track_table.empty else 0
    confirmed_track_count = confirmed_tracks["track_id"].nunique() if not confirmed_tracks.empty else 0
    return [
        "## 评价指标说明",
        "",
        "由于当前数据没有 AIS 真值或人工标注，不能计算严格意义上的准确率、召回率和 F1 值。",
        "本实验采用无监督质量评价：候选数量、连续帧数、直线性、相邻帧最大跳变、最大转向角和平均信噪比。",
        f"筛选流程从 {len(cluster_table)} 个候选簇形成 {track_count} 条候选航迹，最终保留 {confirmed_track_count} 条质量确认航迹。",
        "若后续获得 AIS 或人工标注，才可以进一步计算检测率、虚警率和航迹误差。",
        "",
    ]


def build_analysis_text() -> str:
    # 生成实验结果分析正文
    data_summary = pd.read_csv(TABLE_DIR / "data_summary.csv")
    frame_summary = pd.read_csv(TABLE_DIR / "frame_summary.csv")
    cluster_table = pd.read_csv(RESULT_DIR / "candidate_clusters.csv")
    track_table = pd.read_csv(RESULT_DIR / "candidate_tracks.csv")
    confirmed_tracks = pd.read_csv(RESULT_DIR / "confirmed_tracks.csv")
    track_summary = pd.read_csv(RESULT_DIR / "track_summary.csv")
    display_summary = build_display_summary(track_summary)
    track_review_lines = build_track_review_lines(display_summary)
    time_review_lines = build_time_review_lines(frame_summary)
    metric_review_lines = build_metric_review_lines(cluster_table, track_table, confirmed_tracks)
    confirmed_track_count = confirmed_tracks["track_id"].nunique() if "track_id" in confirmed_tracks else 0
    confirmed_track_label = "确认航迹"
    if not confirmed_tracks.empty and "track_id" in confirmed_tracks:
        confirmed_track_ids = sorted(confirmed_tracks["track_id"].astype(int).unique().tolist())
        confirmed_track_label = "、".join(f"T{track_id}" for track_id in confirmed_track_ids)

    lines = [
        "# 实验结果分析",
        "",
        "## 数据与检测结果",
        "",
        f"原始数据共包含 {get_summary_value(data_summary, '帧数')} 帧、{get_summary_value(data_summary, '原始总点迹数')} 个点迹。",
        (
            f"清洗无效信噪比点后，保留 {get_summary_value(data_summary, '清洗后点迹数')} 个点迹，"
            f"无效信噪比点数为 {get_summary_value(data_summary, '无效信噪比点数')}。"
        ),
        f"按帧执行信噪比和幅度分位筛选后，使用空间-速度-信号强度特征进行 DBSCAN 聚类，共得到 {len(cluster_table)} 个目标候选簇。",
        f"再通过相邻帧最近邻关联、速度一致性、方向一致性、最小持续帧数和形态质量约束，得到 {confirmed_track_count} 条疑似目标航迹。",
        "",
        *time_review_lines,
        *metric_review_lines,
        "## 航迹质量审查",
        "",
        *track_review_lines,
        "",
        "## 图件说明",
        "",
        "- `图1_点迹空间密度`：展示全部点迹在平面坐标中的空间密度，用于说明观测区域和点迹背景分布。",
        "- `图2_点迹空间分布`：展示全部点迹的散点形态，用于辅助理解空间覆盖范围和离散点迹结构。",
        "- `图3_多帧候选检测`：展示确认航迹中的四个代表帧，说明每帧候选点和候选簇如何形成。",
        "- `图4_候选簇与确认航迹`：展示全局候选簇中心和最终确认航迹在观测空间中的相对位置。",
        f"- `图5_确认航迹局部放大`：展示质量确认后 {confirmed_track_label} 在连续帧之间的候选中心移动。",
        "- `图6_候选筛选数量评价`：展示候选簇、候选航迹、长度达标航迹和质量确认航迹的数量变化。",
        "- `图7_航迹质量评价`：展示长度达标候选航迹的直线性、相邻帧跳变和最大转向角。",
        "",
        "## 结果边界",
        "",
        "当前结果应表述为“目标候选”或“疑似目标航迹”，不能表述为已验证船舶目标。",
        "多帧确认采用的是简化最近邻关联，并未接入 AIS 真值、人工标注或完整卡尔曼滤波，因此只能用于课程作业中的方法演示和候选结果分析。",
        "后续若获得 AIS 或人工标注，可进一步计算检测率、虚警率、航迹误差和关联正确率。",
        "",
    ]
    return "\n".join(lines)


def main():
    # 设置控制台编码
    sys.stdout.reconfigure(encoding="utf-8")

    # 生成分析文档
    analysis_text = build_analysis_text()
    ANALYSIS_FILE.write_text(analysis_text, encoding="utf-8")
    print(f"实验结果分析已保存：{ANALYSIS_FILE}")


if __name__ == "__main__":
    # 执行实验分析生成流程
    main()
