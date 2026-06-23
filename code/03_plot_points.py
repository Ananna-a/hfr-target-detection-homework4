import sys

import matplotlib.pyplot as plt
import numpy as np

from hfr_utils import (
    all_frames_to_dataframe,
    configure_chinese_plot,
    ensure_output_dirs,
    load_hfr_data,
    save_figure,
)


# 展示帧来源：覆盖前中后典型时刻
SELECTED_FRAME_ID_LIST = [1, 10, 20]
# 点大小来源：散点图需要避免遮挡
SCATTER_SIZE = 8
# 透明度来源：高密度点迹需要观察重叠区域
SCATTER_ALPHA = 0.65
# 直方图箱数来源：兼顾分布细节和报告可读性
HIST_BIN_COUNT = 40
# 图像宽度来源：报告横向图片排版
FIGURE_WIDTH = 10
# 图像高度来源：报告横向图片排版
FIGURE_HEIGHT = 7


def plot_frame_count(point_table):
    # 绘制每帧点迹数量变化图
    frame_count = point_table.groupby("frame_idx").size()
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    axis.plot(frame_count.index, frame_count.values, marker="o", linewidth=1.8)
    axis.set_title("每帧点迹数量变化")
    axis.set_xlabel("帧编号")
    axis.set_ylabel("点迹数量")
    axis.grid(True, linestyle="--", alpha=0.35)
    return save_figure(figure, "每帧点迹数量变化.png")


def plot_xy_scatter(point_table):
    # 绘制典型帧x-y点迹散点图
    figure, axes = plt.subplots(1, len(SELECTED_FRAME_ID_LIST), figsize=(FIGURE_WIDTH * 1.5, FIGURE_HEIGHT))
    for axis, frame_id in zip(axes, SELECTED_FRAME_ID_LIST):
        frame_table = point_table[point_table["frame_idx"] == frame_id]
        axis.scatter(frame_table["x"], frame_table["y"], s=SCATTER_SIZE, alpha=SCATTER_ALPHA)
        axis.set_title(f"第{frame_id}帧原始点迹")
        axis.set_xlabel("x坐标/km")
        axis.set_ylabel("y坐标/km")
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.set_aspect("equal", adjustable="box")
    return save_figure(figure, "典型帧原始点迹散点图.png")


def plot_lon_lat(point_table):
    # 绘制经纬度空间分布图
    finite_snr_table = point_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["snr"])
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    scatter = axis.scatter(
        finite_snr_table["lon"],
        finite_snr_table["lat"],
        c=finite_snr_table["snr"],
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        cmap="viridis",
    )
    figure.colorbar(scatter, ax=axis, label="信噪比/dB")
    axis.set_title("全部点迹经纬度分布")
    axis.set_xlabel("经度")
    axis.set_ylabel("纬度")
    axis.grid(True, linestyle="--", alpha=0.35)
    return save_figure(figure, "全部点迹经纬度分布.png")


def plot_range_velocity(point_table):
    # 绘制距离速度二维分布图
    finite_snr_table = point_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["snr"])
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    scatter = axis.scatter(
        finite_snr_table["range"],
        finite_snr_table["velocity"],
        c=finite_snr_table["snr"],
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        cmap="plasma",
    )
    figure.colorbar(scatter, ax=axis, label="信噪比/dB")
    axis.set_title("距离-速度点迹分布")
    axis.set_xlabel("距离/km")
    axis.set_ylabel("速度/(km/h)")
    axis.grid(True, linestyle="--", alpha=0.35)
    return save_figure(figure, "距离速度点迹分布.png")


def plot_snr_hist(point_table):
    # 绘制有效信噪比分布直方图
    finite_snr = point_table["snr"].replace([np.inf, -np.inf], np.nan).dropna()
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    axis.hist(finite_snr, bins=HIST_BIN_COUNT, color="#2c7fb8", alpha=0.85)
    axis.set_title("有效信噪比分布")
    axis.set_xlabel("信噪比/dB")
    axis.set_ylabel("点迹数量")
    axis.grid(True, linestyle="--", alpha=0.35)
    return save_figure(figure, "有效信噪比分布.png")


def main():
    # 设置控制台编码和中文绘图
    sys.stdout.reconfigure(encoding="utf-8")
    configure_chinese_plot()
    ensure_output_dirs()

    # 读取点迹并生成图表
    hfr_data = load_hfr_data()
    point_table = all_frames_to_dataframe(hfr_data)
    figure_paths = [
        plot_frame_count(point_table),
        plot_xy_scatter(point_table),
        plot_lon_lat(point_table),
        plot_range_velocity(point_table),
        plot_snr_hist(point_table),
    ]

    # 输出图片路径
    for figure_path in figure_paths:
        print(f"图像已保存：{figure_path}")


if __name__ == "__main__":
    # 执行基础可视化流程
    main()
