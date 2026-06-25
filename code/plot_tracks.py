"""快速绘制V2版确认航迹图"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hfr_config import (
    FIGURE_DIR,
    RESULT_DIR,
    BLUE_MAIN,
    BLUE_SECONDARY,
    RED_STRONG,
    NEUTRAL_DARK,
    NEUTRAL_LIGHT,
    NEUTRAL_MID,
    TRACK_COLOR_LIST,
    configure_plot_style,
    ensure_output_dirs,
    save_figure,
)

TRACK_COLORS = [
    "#B64342", "#2F8F5B", "#9A4D8E", "#0F4D92",
    "#D4752B", "#3B8C8C", "#7B4B8A", "#C25A3A",
    "#4A7BA7", "#8B5E3C", "#5E9E6B", "#A0526B",
]


def plot_global_all_tracks(track_table, track_summary):
    """全局总览：所有确认航迹"""
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # 按直线性排序，好航迹用突出颜色
    top_ids = set(track_summary.head(12)["track_id"].astype(int).tolist())

    for idx, (tid, grp) in enumerate(track_table.groupby("track_id")):
        grp = grp.sort_values("frame_idx")
        if int(tid) in top_ids:
            color = TRACK_COLORS[idx % len(TRACK_COLORS)]
            alpha, lw, z = 0.9, 1.2, 5
            label = f"T{int(tid)}"
        else:
            color = NEUTRAL_LIGHT
            alpha, lw, z = 0.3, 0.4, 1
            label = None
        ax.plot(grp["smooth_x"], grp["smooth_y"], "-", color=color,
                linewidth=lw, alpha=alpha, zorder=z, label=label)
        ax.scatter([grp["smooth_x"].iloc[0]], [grp["smooth_y"].iloc[0]],
                   s=12, color=color, alpha=alpha, zorder=z)

    ax.scatter([0], [0], s=60, marker="^", color=RED_STRONG, zorder=10, label="坐标原点")
    ax.set_xlabel("x / km")
    ax.set_ylabel("y / km")
    ax.set_title(f"全部 {len(track_summary)} 条确认航迹（彩色=Top12）")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=False, fontsize=6, ncol=2, handletextpad=0.3)
    save_figure(fig, "图4_全部确认航迹总览.png")


def plot_top_tracks_grid(track_table, track_summary, top_n=12):
    """Top航迹分格放大图"""
    top_summary = track_summary.head(top_n)
    top_ids = top_summary["track_id"].astype(int).tolist()

    cols = 4
    rows = (top_n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.2), constrained_layout=True)
    flat_axes = axes.ravel()

    for ax_idx, tid in enumerate(top_ids):
        ax = flat_axes[ax_idx]
        grp = track_table[track_table["track_id"] == tid].sort_values("frame_idx")
        st = top_summary[top_summary["track_id"] == tid].iloc[0]

        color = TRACK_COLORS[ax_idx % len(TRACK_COLORS)]
        ax.plot(grp["smooth_x"], grp["smooth_y"], "-o", color=color,
                linewidth=1.5, markersize=3, zorder=3)
        ax.scatter([grp["smooth_x"].iloc[0]], [grp["smooth_y"].iloc[0]],
                   s=50, marker="o", color=RED_STRONG, zorder=5, label="起点")
        ax.scatter([grp["smooth_x"].iloc[-1]], [grp["smooth_y"].iloc[-1]],
                   s=50, marker="s", color=NEUTRAL_DARK, zorder=5, label="终点")

        # 加方向箭头
        for k in range(len(grp) - 1):
            ax.annotate("", xy=(grp["smooth_x"].iloc[k+1], grp["smooth_y"].iloc[k+1]),
                        xytext=(grp["smooth_x"].iloc[k], grp["smooth_y"].iloc[k]),
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 0.6,
                                    "shrinkA": 3, "shrinkB": 3})

        title = f"T{tid} | {int(st['frame_count'])}帧 | 直线性{st['straightness']:.2f} | SNR{st['mean_snr']:.1f}"
        ax.set_title(title, fontsize=7, pad=3)
        ax.set_xlabel("x/km", fontsize=6)
        ax.set_ylabel("y/km", fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        if ax_idx == 0:
            ax.legend(fontsize=5, frameon=False, loc="best")

    # 隐藏多余子图
    for ax_idx in range(top_n, len(flat_axes)):
        flat_axes[ax_idx].axis("off")

    fig.suptitle("综合评分 Top 12 确认航迹放大图", fontsize=12, y=1.01)
    save_figure(fig, "图5_Top12航迹放大图.png")


def plot_top_tracks_with_raw(track_table, track_summary, top_n=12):
    """Top航迹：KF平滑轨迹 + 原始量测点"""
    top_summary = track_summary.head(top_n)
    top_ids = top_summary["track_id"].astype(int).tolist()

    cols = 3
    rows = (top_n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4.5), constrained_layout=True)
    flat_axes = axes.ravel()

    for ax_idx, tid in enumerate(top_ids):
        ax = flat_axes[ax_idx]
        grp = track_table[track_table["track_id"] == tid].sort_values("frame_idx")
        st = top_summary[top_summary["track_id"] == tid].iloc[0]

        color = TRACK_COLORS[ax_idx % len(TRACK_COLORS)]

        # 原始量测点
        meas = grp[["meas_x", "meas_y"]].dropna()
        if len(meas) > 0:
            ax.scatter(meas["meas_x"], meas["meas_y"], s=20, color=color,
                       alpha=0.35, edgecolors="none", label="原始量测", zorder=1)

        # KF平滑轨迹
        ax.plot(grp["smooth_x"], grp["smooth_y"], "-", color=color,
                linewidth=1.8, alpha=0.85, label="KF平滑", zorder=3)

        # 起终点
        ax.scatter([grp["smooth_x"].iloc[0]], [grp["smooth_y"].iloc[0]],
                   s=60, marker="o", facecolor=RED_STRONG, edgecolors="white",
                   linewidth=0.8, zorder=5)
        ax.scatter([grp["smooth_x"].iloc[-1]], [grp["smooth_y"].iloc[-1]],
                   s=60, marker="s", facecolor=NEUTRAL_DARK, edgecolors="white",
                   linewidth=0.8, zorder=5)

        ax.set_title(f"T{tid} | {int(st.frame_count)}帧 | 位移{st.displacement:.1f}km | 直线性{st.straightness:.2f}",
                     fontsize=7.5, pad=3)
        ax.set_xlabel("x / km", fontsize=6.5)
        ax.set_ylabel("y / km", fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

        if ax_idx == 0:
            handles = [
                plt.Line2D([], [], color=color, linewidth=1.8, label="KF平滑"),
                plt.Line2D([], [], marker="o", color="w", markerfacecolor=color, alpha=0.35,
                           markersize=6, label="原始量测", linestyle=""),
            ]
            ax.legend(handles=handles, fontsize=5.5, frameon=False, loc="best")

    for ax_idx in range(top_n, len(flat_axes)):
        flat_axes[ax_idx].axis("off")

    fig.suptitle(f"Top 12 航迹详情：KF平滑轨迹 + 原始雷达量测点", fontsize=12, y=1.01)
    save_figure(fig, "图5b_Top12航迹KF与原始量测.png")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ensure_output_dirs()
    configure_plot_style()

    track_table = pd.read_csv(RESULT_DIR / "confirmed_tracks.csv")
    track_summary = pd.read_csv(RESULT_DIR / "track_summary.csv")

    print(f"加载确认航迹: {track_summary['track_id'].nunique()} 条")
    print(f"Track Table 列: {list(track_table.columns)}")

    # 1. 全局总览
    plot_global_all_tracks(track_table, track_summary)

    # 2. Top12 放大格
    plot_top_tracks_grid(track_table, track_summary)

    # 3. Top12 KF+原始量测
    plot_top_tracks_with_raw(track_table, track_summary)

    print(f"\n图片已保存至: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
