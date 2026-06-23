import sys

import pandas as pd

from hfr_config import PROJECT_ROOT, RESULT_DIR, TABLE_DIR


# 分析文档路径来源：报告实验结果章节可直接引用
ANALYSIS_FILE = PROJECT_ROOT / "report" / "实验结果分析.md"


def dataframe_to_markdown(data_frame: pd.DataFrame) -> list[str]:
    # 将表格转换为Markdown文本
    header_line = "| " + " | ".join(data_frame.columns) + " |"
    split_line = "| " + " | ".join(["---"] * len(data_frame.columns)) + " |"
    body_lines = []
    for row_data in data_frame.to_dict(orient="records"):
        body_lines.append("| " + " | ".join(str(row_data[column_name]) for column_name in data_frame.columns) + " |")
    return [header_line, split_line, *body_lines]


def build_analysis_text() -> str:
    # 生成实验结果分析正文
    data_summary = pd.read_csv(TABLE_DIR / "data_summary.csv")
    cluster_table = pd.read_csv(RESULT_DIR / "candidate_clusters.csv")
    confirmed_tracks = pd.read_csv(RESULT_DIR / "confirmed_tracks.csv")
    track_summary = pd.read_csv(RESULT_DIR / "track_summary.csv")

    display_summary = track_summary[
        [
            "track_id",
            "frame_count",
            "start_frame",
            "end_frame",
            "mean_snr",
            "mean_velocity",
            "mean_point_count",
            "displacement",
        ]
    ].copy()
    integer_columns = ["track_id", "frame_count", "start_frame", "end_frame"]
    for column_name in integer_columns:
        display_summary[column_name] = display_summary[column_name].astype(int)
    for column_name in ["mean_snr", "mean_velocity", "mean_point_count", "displacement"]:
        display_summary[column_name] = display_summary[column_name].round(2)

    best_track = display_summary.sort_values(["frame_count", "mean_snr"], ascending=False).iloc[0]
    lines = [
        "# 实验结果分析",
        "",
        "## 数据与检测结果",
        "",
        f"整理后数据共包含 {int(data_summary[data_summary['指标'] == '帧数']['数值'].iloc[0])} 帧、{int(data_summary[data_summary['指标'] == '总点迹数']['数值'].iloc[0])} 个点迹。",
        f"采用信噪比和幅度分位阈值进行强点筛选后，进一步通过空间-速度特征聚类得到 {len(cluster_table)} 个目标候选簇。",
        f"多帧最近邻关联和最小持续帧数筛选后，保留 {confirmed_tracks['track_id'].nunique()} 条确认疑似航迹。",
        "",
        "## 确认航迹摘要",
        "",
        *dataframe_to_markdown(display_summary),
        "",
        "## 结果解读",
        "",
        f"持续时间最长的疑似航迹为 T{int(best_track['track_id'])}，从第 {int(best_track['start_frame'])} 帧持续到第 {int(best_track['end_frame'])} 帧，共 {int(best_track['frame_count'])} 帧。",
        f"该航迹平均信噪比为 {best_track['mean_snr']:.2f} dB，平均径向速度为 {best_track['mean_velocity']:.2f} km/h，候选中心平面位移为 {best_track['displacement']:.2f} km。",
        "这说明该目标候选在多帧中具有较好的空间连续性和信号稳定性，可作为主要检测结果展示。",
        "",
        "需要注意，数据中的速度是径向多普勒速度，航迹图中的位移是候选中心在局部平面坐标中的变化，二者不能直接等同。",
        "由于当前数据未提供人工真值标签，结果应表述为“疑似目标航迹”或“目标候选航迹”，不应表述为已验证船舶目标。",
        "",
        "## 图件说明",
        "",
        "- `图1_点迹数据概览`：展示点迹空间密度和距离-径向速度分布，说明数据背景和多普勒维度特征。",
        "- `图2_单帧候选检测`：展示第42帧从强点筛选到候选簇中心提取的过程。",
        "- `图3_多帧确认航迹`：展示经过多帧关联后保留的稳定疑似目标航迹。",
        "",
        "## 局限性",
        "",
        "本实验基于已生成的点迹数据，不能替代原始回波级别的杂波抑制和CFAR处理。",
        "后续若获得人工标注或AIS船舶轨迹，可进一步计算检测率、虚警率和航迹误差。",
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
