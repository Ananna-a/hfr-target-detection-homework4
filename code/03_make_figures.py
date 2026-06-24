import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from hfr_config import (
    BLUE_MAIN,
    DISPLAY_FRAME_ID,
    FIGURE_DIR,
    NEUTRAL_BLACK,
    NEUTRAL_DARK,
    RED_STRONG,
    RESULT_DIR,
    TABLE_DIR,
    configure_plot_style,
    ensure_output_dirs,
    load_point_table,
    save_figure,
)


# 空间密度网格数来源：全局点迹分布展示
SPATIAL_GRIDSIZE = 62
# 全局散点大小来源：全部点迹数量较多
GLOBAL_POINT_SIZE = 3
# 全局散点透明度来源：兼顾密集区和稀疏区
GLOBAL_POINT_ALPHA = 0.34
# 雷达站标记大小来源：空间坐标原点提示
RADAR_MARKER_SIZE = 42
# 单帧背景点大小来源：保证报告缩放后仍可辨认
BACKGROUND_POINT_SIZE = 12
# 单帧背景透明度来源：保留背景结构
BACKGROUND_ALPHA = 0.42
# 强点大小来源：突出信噪比和幅度筛选结果
STRONG_POINT_SIZE = 22
# 候选中心大小来源：标记聚类中心
CENTER_POINT_SIZE = 26
# 局部视窗边距来源：给标注和箭头留白
VIEW_PADDING_KM = 6.0
# 候选簇标注偏移来源：避免遮挡中心点
CLUSTER_LABEL_OFFSET_POINTS = (5, 5)


def read_result_table(file_name: str) -> pd.DataFrame:
    # 读取结果表
    return pd.read_csv(RESULT_DIR / file_name)


def set_equal_axis(axis: plt.Axes) -> None:
    # 设置平面坐标轴样式
    axis.set_xlabel("x / km")
    axis.set_ylabel("y / km")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.38)


def plot_spatial_density(point_table: pd.DataFrame) -> None:
    # 绘制全局点迹空间密度图
    finite_table = point_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
    figure, axis = plt.subplots(figsize=(4.8, 4.25), constrained_layout=True)
    density_plot = axis.hexbin(
        finite_table["x"],
        finite_table["y"],
        gridsize=SPATIAL_GRIDSIZE,
        cmap="Blues",
        norm=LogNorm(),
        mincnt=1,
        linewidths=0,
    )
    axis.set_title("点迹空间密度", pad=5)
    set_equal_axis(axis)
    colorbar = figure.colorbar(density_plot, ax=axis, label="点迹数（对数色标）", fraction=0.048, pad=0.025)
    colorbar.outline.set_linewidth(0.5)
    save_figure(figure, "图1_点迹空间密度.png")


def plot_spatial_distribution(point_table: pd.DataFrame) -> None:
    # 绘制全局点迹空间散点分布图
    finite_table = point_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
    figure, axis = plt.subplots(figsize=(4.8, 4.25), constrained_layout=True)
    axis.scatter(
        finite_table["x"],
        finite_table["y"],
        s=GLOBAL_POINT_SIZE,
        color=NEUTRAL_DARK,
        alpha=GLOBAL_POINT_ALPHA,
        linewidths=0,
        rasterized=True,
    )
    axis.scatter(
        [0],
        [0],
        s=RADAR_MARKER_SIZE,
        marker="^",
        color=RED_STRONG,
        linewidths=0,
        zorder=5,
        label="坐标原点",
    )
    axis.set_title("点迹空间分布", pad=5)
    set_equal_axis(axis)
    axis.legend(loc="upper left", frameon=False, handletextpad=0.45)
    save_figure(figure, "图2_点迹空间分布.png")


def get_candidate_points(frame_strong_points: pd.DataFrame, frame_clusters: pd.DataFrame) -> pd.DataFrame:
    # 提取通过聚类确认的候选点
    accepted_cluster_ids = set(frame_clusters["cluster_id"].astype(int).tolist())
    return frame_strong_points[frame_strong_points["cluster_id"].isin(accepted_cluster_ids)]


def set_detection_view(axis: plt.Axes, frame_table: pd.DataFrame, frame_clusters: pd.DataFrame) -> None:
    # 设置单帧检测图展示范围
    if frame_clusters.empty:
        x_min, x_max = frame_table["x"].min(), frame_table["x"].max()
        y_min, y_max = frame_table["y"].min(), frame_table["y"].max()
    else:
        x_min, x_max = frame_clusters["center_x"].min(), frame_clusters["center_x"].max()
        y_min, y_max = frame_clusters["center_y"].min(), frame_clusters["center_y"].max()
    axis.set_xlim(x_min - VIEW_PADDING_KM * 1.55, x_max + VIEW_PADDING_KM * 1.55)
    axis.set_ylim(y_min - VIEW_PADDING_KM * 1.45, y_max + VIEW_PADDING_KM * 1.45)


def plot_single_frame_detection(
    point_table: pd.DataFrame,
    strong_point_table: pd.DataFrame,
    cluster_table: pd.DataFrame,
) -> None:
    # 绘制单帧候选检测结果图
    frame_table = point_table[point_table["frame_idx"] == DISPLAY_FRAME_ID]
    frame_strong_points = strong_point_table[strong_point_table["frame_idx"] == DISPLAY_FRAME_ID]
    frame_clusters = cluster_table[cluster_table["frame_idx"] == DISPLAY_FRAME_ID]
    frame_candidate_points = get_candidate_points(frame_strong_points, frame_clusters)

    figure, axis = plt.subplots(figsize=(4.9, 4.1), constrained_layout=True)
    axis.scatter(
        frame_table["x"],
        frame_table["y"],
        s=BACKGROUND_POINT_SIZE,
        color=NEUTRAL_DARK,
        alpha=BACKGROUND_ALPHA,
        linewidths=0,
        label="原始点迹",
    )
    axis.scatter(
        frame_candidate_points["x"],
        frame_candidate_points["y"],
        s=STRONG_POINT_SIZE,
        color=BLUE_MAIN,
        alpha=0.9,
        linewidths=0,
        label="候选点",
    )

    sorted_clusters = frame_clusters.sort_values("center_y", ascending=False)
    for cluster_order, cluster_row in enumerate(sorted_clusters.itertuples(), start=1):
        # 标注每个候选簇
        axis.scatter(
            [cluster_row.center_x],
            [cluster_row.center_y],
            s=CENTER_POINT_SIZE,
            marker="o",
            color=RED_STRONG,
            linewidths=0,
            zorder=6,
        )
        axis.annotate(
            f"C{cluster_order}",
            (cluster_row.center_x, cluster_row.center_y),
            xytext=CLUSTER_LABEL_OFFSET_POINTS,
            textcoords="offset points",
            fontsize=7,
            color=NEUTRAL_BLACK,
            zorder=7,
        )

    axis.set_title(f"第{DISPLAY_FRAME_ID}帧候选簇提取", pad=5)
    set_equal_axis(axis)
    set_detection_view(axis, frame_table, frame_clusters)
    axis.scatter([], [], s=CENTER_POINT_SIZE, marker="o", color=RED_STRONG, label="候选中心")
    axis.legend(loc="upper right", frameon=False, handletextpad=0.45)
    save_figure(figure, "图3_单帧候选检测.png")


def main():
    # 设置控制台编码和绘图风格
    sys.stdout.reconfigure(encoding="utf-8")
    ensure_output_dirs()
    configure_plot_style()

    # 读取数据和检测结果
    point_table = load_point_table()
    strong_point_table = pd.read_csv(TABLE_DIR / "strong_points.csv")
    cluster_table = read_result_table("candidate_clusters.csv")

    # 生成三张单图
    plot_spatial_density(point_table)
    plot_spatial_distribution(point_table)
    plot_single_frame_detection(point_table, strong_point_table, cluster_table)

    # 输出图片目录
    print(f"主图已保存：{FIGURE_DIR}")


if __name__ == "__main__":
    # 执行报告主图生成流程
    main()
