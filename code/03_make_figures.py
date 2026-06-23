import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hfr_config import (
    BLUE_MAIN,
    BLUE_SECONDARY,
    DISPLAY_FRAME_ID,
    FIGURE_DIR,
    NEUTRAL_BLACK,
    NEUTRAL_LIGHT,
    NEUTRAL_MID,
    RED_STRONG,
    RESULT_DIR,
    TABLE_DIR,
    TRACK_COLOR_LIST,
    configure_plot_style,
    ensure_output_dirs,
    load_point_table,
    save_figure,
)


# 空间密度网格数来源：兼顾细节和可读性
SPATIAL_GRIDSIZE = 56
# 距离速度密度网格数来源：体现R-D域分布
RANGE_VELOCITY_GRIDSIZE = 58
# 原始点透明度来源：作为背景信息弱化显示
BACKGROUND_ALPHA = 0.28
# 候选点大小来源：比背景点更醒目
CANDIDATE_POINT_SIZE = 13
# 中心点大小来源：突出候选中心但不遮挡
CENTER_POINT_SIZE = 38


def add_panel_label(axis, label: str) -> None:
    # 添加Nature风格小写面板编号
    axis.text(-0.11, 1.05, label, transform=axis.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")


def read_result_table(file_name: str) -> pd.DataFrame:
    # 读取结果表
    return pd.read_csv(RESULT_DIR / file_name)


def plot_data_overview(point_table: pd.DataFrame) -> None:
    # 绘制数据空间分布和距离速度分布
    finite_table = point_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y", "range", "velocity"])
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)

    spatial_plot = axes[0].hexbin(
        finite_table["x"],
        finite_table["y"],
        gridsize=SPATIAL_GRIDSIZE,
        cmap="Blues",
        mincnt=1,
        linewidths=0,
    )
    axes[0].set_title("点迹空间密度")
    axes[0].set_xlabel("x / km")
    axes[0].set_ylabel("y / km")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.35)
    add_panel_label(axes[0], "a")
    figure.colorbar(spatial_plot, ax=axes[0], label="点迹数", fraction=0.046, pad=0.03)

    range_velocity_plot = axes[1].hexbin(
        finite_table["range"],
        finite_table["velocity"],
        gridsize=RANGE_VELOCITY_GRIDSIZE,
        cmap="Greys",
        mincnt=1,
        linewidths=0,
    )
    axes[1].axhline(0, color=NEUTRAL_MID, linewidth=0.8, linestyle="--", alpha=0.8)
    axes[1].set_title("距离-径向速度分布")
    axes[1].set_xlabel("距离 / km")
    axes[1].set_ylabel("径向速度 / (km h$^{-1}$)")
    axes[1].grid(True, alpha=0.35)
    add_panel_label(axes[1], "b")
    figure.colorbar(range_velocity_plot, ax=axes[1], label="点迹数", fraction=0.046, pad=0.03)

    save_figure(figure, "图1_点迹数据概览.png")


def plot_detection_process(point_table: pd.DataFrame, strong_point_table: pd.DataFrame, cluster_table: pd.DataFrame) -> None:
    # 绘制单帧目标候选检测过程
    frame_table = point_table[point_table["frame_idx"] == DISPLAY_FRAME_ID]
    frame_strong_points = strong_point_table[strong_point_table["frame_idx"] == DISPLAY_FRAME_ID]
    frame_clusters = cluster_table[cluster_table["frame_idx"] == DISPLAY_FRAME_ID]
    accepted_cluster_ids = set(frame_clusters["cluster_id"].astype(int).tolist())
    frame_candidate_points = frame_strong_points[frame_strong_points["cluster_id"].isin(accepted_cluster_ids)]

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    for axis in axes:
        axis.scatter(frame_table["x"], frame_table["y"], s=7, color=NEUTRAL_LIGHT, alpha=BACKGROUND_ALPHA, linewidths=0)
        axis.set_xlabel("x / km")
        axis.set_ylabel("y / km")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.35)
        if not frame_clusters.empty:
            axis.set_xlim(frame_clusters["center_x"].min() - 42, frame_clusters["center_x"].max() + 42)
            axis.set_ylim(frame_clusters["center_y"].min() - 42, frame_clusters["center_y"].max() + 42)

    axes[0].scatter(frame_strong_points["x"], frame_strong_points["y"], s=10, color=BLUE_SECONDARY, alpha=0.72, linewidths=0)
    axes[0].set_title(f"第{DISPLAY_FRAME_ID}帧强点筛选")
    axes[0].text(
        0.03,
        0.96,
        f"强点数：{len(frame_strong_points)}",
        transform=axes[0].transAxes,
        va="top",
        fontsize=7,
        bbox={"fc": "white", "ec": "#cccccc", "lw": 0.4, "boxstyle": "round,pad=0.25"},
    )
    add_panel_label(axes[0], "a")

    axes[1].scatter(frame_candidate_points["x"], frame_candidate_points["y"], s=CANDIDATE_POINT_SIZE, color=BLUE_MAIN, alpha=0.82, linewidths=0)
    axes[1].scatter(
        frame_clusters["center_x"],
        frame_clusters["center_y"],
        s=CENTER_POINT_SIZE,
        marker="x",
        color=RED_STRONG,
        linewidths=1.4,
    )
    axes[1].set_title("聚类后的候选目标")
    axes[1].text(
        0.03,
        0.96,
        f"候选簇数：{len(frame_clusters)}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=7,
        bbox={"fc": "white", "ec": "#cccccc", "lw": 0.4, "boxstyle": "round,pad=0.25"},
    )
    add_panel_label(axes[1], "b")

    save_figure(figure, "图2_单帧候选检测.png")


def plot_confirmed_tracks(cluster_table: pd.DataFrame, confirmed_tracks: pd.DataFrame, track_summary: pd.DataFrame) -> None:
    # 绘制多帧确认航迹
    figure, axis = plt.subplots(figsize=(5.1, 4.2), constrained_layout=True)
    if confirmed_tracks.empty:
        axis.text(0.5, 0.5, "未形成确认航迹", transform=axis.transAxes, ha="center", va="center")
        save_figure(figure, "图3_多帧确认航迹.png")
        return

    x_min = confirmed_tracks["center_x"].min() - 18
    x_max = confirmed_tracks["center_x"].max() + 18
    y_min = confirmed_tracks["center_y"].min() - 16
    y_max = confirmed_tracks["center_y"].max() + 16
    background_clusters = cluster_table[
        (cluster_table["center_x"] >= x_min)
        & (cluster_table["center_x"] <= x_max)
        & (cluster_table["center_y"] >= y_min)
        & (cluster_table["center_y"] <= y_max)
    ]

    axis.scatter(background_clusters["center_x"], background_clusters["center_y"], s=12, color=NEUTRAL_LIGHT, alpha=0.45, linewidths=0)

    for color_idx, (track_id, track_table) in enumerate(confirmed_tracks.groupby("track_id")):
        ordered_track = track_table.sort_values("frame_idx")
        track_color = TRACK_COLOR_LIST[color_idx % len(TRACK_COLOR_LIST)]
        start_row = ordered_track.iloc[0]
        end_row = ordered_track.iloc[-1]
        summary_row = track_summary[track_summary["track_id"] == track_id].iloc[0]
        axis.plot(
            ordered_track["center_x"],
            ordered_track["center_y"],
            color=track_color,
            linewidth=1.35,
            marker="o",
            markersize=2.8,
            alpha=0.95,
        )
        axis.scatter(start_row["center_x"], start_row["center_y"], s=32, facecolors="white", edgecolors=track_color, linewidths=1.0, zorder=4)
        axis.scatter(end_row["center_x"], end_row["center_y"], s=34, marker="s", color=track_color, zorder=4)
        axis.annotate(
            "",
            xy=(end_row["center_x"], end_row["center_y"]),
            xytext=(ordered_track.iloc[-2]["center_x"], ordered_track.iloc[-2]["center_y"]),
            arrowprops={"arrowstyle": "->", "color": track_color, "lw": 1.1},
        )
        axis.annotate(
            f"T{int(track_id)}  F{int(summary_row['start_frame'])}-F{int(summary_row['end_frame'])}",
            (end_row["center_x"], end_row["center_y"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color=NEUTRAL_BLACK,
            bbox={"fc": "white", "ec": "#cccccc", "lw": 0.35, "boxstyle": "round,pad=0.18", "alpha": 0.9},
        )

    axis.set_title("多帧确认的疑似目标航迹")
    axis.set_xlabel("x / km")
    axis.set_ylabel("y / km")
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.35)
    axis.scatter([], [], s=12, color=NEUTRAL_LIGHT, label="候选中心")
    axis.scatter([], [], s=32, facecolors="white", edgecolors=NEUTRAL_BLACK, linewidths=1.0, label="起点")
    axis.scatter([], [], s=34, marker="s", color=NEUTRAL_BLACK, label="终点")
    axis.legend(loc="upper right", frameon=False)

    save_figure(figure, "图3_多帧确认航迹.png")


def main():
    # 设置控制台编码和绘图风格
    sys.stdout.reconfigure(encoding="utf-8")
    ensure_output_dirs()
    configure_plot_style()

    # 读取数据和结果
    point_table = load_point_table()
    strong_point_table = pd.read_csv(TABLE_DIR / "strong_points.csv")
    cluster_table = read_result_table("candidate_clusters.csv")
    confirmed_tracks = read_result_table("confirmed_tracks.csv")
    track_summary = read_result_table("track_summary.csv")

    # 生成三张主图
    plot_data_overview(point_table)
    plot_detection_process(point_table, strong_point_table, cluster_table)
    plot_confirmed_tracks(cluster_table, confirmed_tracks, track_summary)

    # 输出图片目录
    print(f"主图已保存：{FIGURE_DIR}")


if __name__ == "__main__":
    # 执行报告主图生成流程
    main()
