import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from hfr_config import (
    BLUE_MAIN,
    BLUE_SECONDARY,
    DISPLAY_FRAME_ID,
    FIGURE_DIR,
    MAX_DIRECTION_CHANGE_DEG,
    MAX_MEAN_TRACK_STEP_KM,
    MAX_TRACK_STEP_KM,
    MIN_TRACK_LENGTH,
    MIN_TRACK_STRAIGHTNESS,
    NEUTRAL_BLACK,
    NEUTRAL_DARK,
    NEUTRAL_MID,
    RED_STRONG,
    RESULT_DIR,
    TABLE_DIR,
    MIN_DIRECTION_STEP_KM,
    TRACK_COLOR_LIST,
    TRACK_MARKER_LIST,
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
# 航迹中心点大小来源：突出每帧候选中心
TRACK_POINT_SIZE = 34
# 航迹背景点大小来源：提供局部环境参照
TRACK_BACKGROUND_POINT_SIZE = 8
# 航迹背景透明度来源：避免压住轨迹线
TRACK_BACKGROUND_ALPHA = 0.12
# 航迹线宽来源：报告缩放后保持可辨认
TRACK_LINE_WIDTH = 1.4
# 航迹横向视窗边距来源：保留起终点和帧号标注
TRACK_VIEW_PADDING_X_KM = 1.45
# 航迹纵向视窗边距来源：减少短航迹图面堆叠
TRACK_VIEW_PADDING_Y_KM = 1.35
# 航迹坐标刻度数来源：减少局部图拥挤感
TRACK_AXIS_TICK_COUNT = 4
# 多帧候选展示帧数来源：报告拼图可读性
MULTI_FRAME_PANEL_COUNT = 4
# 多帧候选拼图列数来源：报告版面
MULTI_FRAME_GRID_COLS = 2
# 多帧候选拼图尺寸来源：保持四个子图等大并压缩中部空白
MULTI_FRAME_FIGURE_SIZE = (6.35, 5.9)
# 多帧候选坐标刻度数来源：保持四个子图刻度一致
MULTI_FRAME_AXIS_TICK_COUNT = 4
# 多帧候选左边距来源：容纳纵轴标签
MULTI_FRAME_LEFT_MARGIN = 0.085
# 多帧候选右边距来源：保持左右对称
MULTI_FRAME_RIGHT_MARGIN = 0.985
# 多帧候选下边距来源：容纳横轴标签
MULTI_FRAME_BOTTOM_MARGIN = 0.075
# 多帧候选上边距来源：容纳统一图例
MULTI_FRAME_TOP_MARGIN = 0.895
# 多帧候选横向间距来源：减少左右面板中部空白
MULTI_FRAME_WSPACE = 0.08
# 多帧候选纵向间距来源：减少上下面板空白
MULTI_FRAME_HSPACE = 0.22
# 多帧候选图例高度来源：固定图例位置
MULTI_FRAME_LEGEND_Y = 0.985
# 指标柱状图宽度来源：分类数量展示
METRIC_BAR_WIDTH = 0.58
# 指标文字偏移来源：避免贴住柱顶
METRIC_LABEL_PADDING = 0.04
# 无有效转向角标注高度来源：避免贴住横轴
TURN_NOTE_Y_DEG = 8.0
# 无有效转向角标注文字来源：说明短位移不参与角度计算
TURN_NOTE_TEXT = "短步长未计"
# 候选簇中心点大小来源：全局候选分布展示
CLUSTER_CENTER_POINT_SIZE = 18
# 确认航迹全局线宽来源：突出最终保留航迹
GLOBAL_TRACK_LINE_WIDTH = 1.2

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


def get_track_frame_ids(confirmed_tracks: pd.DataFrame) -> list[int]:
    # 选择确认航迹中的代表帧
    if confirmed_tracks.empty:
        return [DISPLAY_FRAME_ID]
    selected_frames = []
    track_groups = list(confirmed_tracks.sort_values(["track_id", "frame_idx"]).groupby("track_id"))
    for _, track_group in track_groups:
        # 优先展示每条确认航迹的起始帧
        track_frames = track_group["frame_idx"].astype(int).sort_values().unique().tolist()
        selected_frames.append(track_frames[0])
    if len(selected_frames) < MULTI_FRAME_PANEL_COUNT:
        # 剩余位置展示最长确认航迹的结束帧
        longest_track = max(track_groups, key=lambda item: len(item[1]))[1]
        longest_frames = longest_track["frame_idx"].astype(int).sort_values().unique().tolist()
        selected_frames.append(longest_frames[-1])
    unique_frames = sorted(set(selected_frames))
    return unique_frames[:MULTI_FRAME_PANEL_COUNT]


def get_multi_frame_view_limits(
    point_table: pd.DataFrame,
    cluster_table: pd.DataFrame,
    frame_ids: list[int],
) -> tuple[float, float, float, float]:
    # 计算多帧拼图统一展示范围
    selected_clusters = cluster_table[cluster_table["frame_idx"].isin(frame_ids)]
    if selected_clusters.empty:
        selected_points = point_table[point_table["frame_idx"].isin(frame_ids)]
        x_min, x_max = selected_points["x"].min(), selected_points["x"].max()
        y_min, y_max = selected_points["y"].min(), selected_points["y"].max()
    else:
        x_min, x_max = selected_clusters["center_x"].min(), selected_clusters["center_x"].max()
        y_min, y_max = selected_clusters["center_y"].min(), selected_clusters["center_y"].max()
    return (
        x_min - VIEW_PADDING_KM * 1.55,
        x_max + VIEW_PADDING_KM * 1.55,
        y_min - VIEW_PADDING_KM * 1.45,
        y_max + VIEW_PADDING_KM * 1.45,
    )


def set_multi_frame_ticks(axis: plt.Axes, view_limits: tuple[float, float, float, float]) -> None:
    # 设置多帧拼图统一坐标刻度
    x_ticks = np.linspace(view_limits[0], view_limits[1], MULTI_FRAME_AXIS_TICK_COUNT)
    y_ticks = np.linspace(view_limits[2], view_limits[3], MULTI_FRAME_AXIS_TICK_COUNT)
    axis.set_xticks(x_ticks)
    axis.set_yticks(y_ticks)
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))


def set_multi_frame_outer_labels(axis: plt.Axes, axis_index: int) -> None:
    # 设置多帧拼图外侧坐标标签
    is_left_column = axis_index % MULTI_FRAME_GRID_COLS == 0
    is_bottom_row = axis_index >= MULTI_FRAME_GRID_COLS
    axis.set_ylabel("y / km" if is_left_column else "")
    axis.set_xlabel("x / km" if is_bottom_row else "")
    axis.tick_params(labelleft=is_left_column, labelbottom=is_bottom_row)


def draw_frame_detection(
    axis: plt.Axes,
    point_table: pd.DataFrame,
    strong_point_table: pd.DataFrame,
    cluster_table: pd.DataFrame,
    frame_id: int,
    show_labels: bool,
    view_limits: tuple[float, float, float, float] | None = None,
) -> None:
    # 绘制指定帧候选检测结果
    frame_table = point_table[point_table["frame_idx"] == frame_id]
    frame_strong_points = strong_point_table[strong_point_table["frame_idx"] == frame_id]
    frame_clusters = cluster_table[cluster_table["frame_idx"] == frame_id]
    frame_candidate_points = get_candidate_points(frame_strong_points, frame_clusters)

    axis.scatter(
        frame_table["x"],
        frame_table["y"],
        s=BACKGROUND_POINT_SIZE,
        color=NEUTRAL_DARK,
        alpha=BACKGROUND_ALPHA,
        linewidths=0,
        label="原始点迹" if show_labels else None,
    )
    axis.scatter(
        frame_candidate_points["x"],
        frame_candidate_points["y"],
        s=STRONG_POINT_SIZE,
        color=BLUE_MAIN,
        alpha=0.9,
        linewidths=0,
        label="候选点" if show_labels else None,
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
            label="候选中心" if show_labels and cluster_order == 1 else None,
        )
        axis.annotate(
            f"C{cluster_order}",
            (cluster_row.center_x, cluster_row.center_y),
            xytext=CLUSTER_LABEL_OFFSET_POINTS,
            textcoords="offset points",
            fontsize=6.4,
            color=NEUTRAL_BLACK,
            zorder=7,
        )

    axis.set_title(f"F{frame_id}", pad=4)
    set_equal_axis(axis)
    if view_limits is None:
        set_detection_view(axis, frame_table, frame_clusters)
    else:
        axis.set_xlim(view_limits[0], view_limits[1])
        axis.set_ylim(view_limits[2], view_limits[3])


def plot_multi_frame_detection(
    point_table: pd.DataFrame,
    strong_point_table: pd.DataFrame,
    cluster_table: pd.DataFrame,
    confirmed_tracks: pd.DataFrame,
) -> None:
    # 绘制多帧候选检测拼图
    frame_ids = get_track_frame_ids(confirmed_tracks)
    figure, axes = plt.subplots(
        nrows=2,
        ncols=MULTI_FRAME_GRID_COLS,
        figsize=MULTI_FRAME_FIGURE_SIZE,
        constrained_layout=False,
    )
    figure.subplots_adjust(
        left=MULTI_FRAME_LEFT_MARGIN,
        right=MULTI_FRAME_RIGHT_MARGIN,
        bottom=MULTI_FRAME_BOTTOM_MARGIN,
        top=MULTI_FRAME_TOP_MARGIN,
        wspace=MULTI_FRAME_WSPACE,
        hspace=MULTI_FRAME_HSPACE,
    )
    flat_axes = axes.ravel()
    view_limits = get_multi_frame_view_limits(point_table, cluster_table, frame_ids)
    for axis_index, axis in enumerate(flat_axes):
        # 填充代表帧子图
        if axis_index >= len(frame_ids):
            axis.axis("off")
            continue
        draw_frame_detection(
            axis,
            point_table,
            strong_point_table,
            cluster_table,
            frame_ids[axis_index],
            show_labels=axis_index == 0,
            view_limits=view_limits,
        )
        set_multi_frame_ticks(axis, view_limits)
        set_multi_frame_outer_labels(axis, axis_index)
    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, MULTI_FRAME_LEGEND_Y),
        frameon=False,
    )
    save_figure(figure, "图3_多帧候选检测.png")


def plot_global_candidate_clusters(
    point_table: pd.DataFrame,
    cluster_table: pd.DataFrame,
    confirmed_tracks: pd.DataFrame,
) -> None:
    # 绘制全局候选簇与确认航迹图
    finite_table = point_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
    figure, axis = plt.subplots(figsize=(4.9, 4.25), constrained_layout=True)
    axis.scatter(
        finite_table["x"],
        finite_table["y"],
        s=GLOBAL_POINT_SIZE,
        color=NEUTRAL_DARK,
        alpha=GLOBAL_POINT_ALPHA,
        linewidths=0,
        rasterized=True,
        label="全体点迹",
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
    axis.scatter(
        cluster_table["center_x"],
        cluster_table["center_y"],
        s=CLUSTER_CENTER_POINT_SIZE,
        color=BLUE_MAIN,
        alpha=0.72,
        linewidths=0,
        label="候选簇中心",
    )
    if not confirmed_tracks.empty:
        for track_order, (track_id, track_group) in enumerate(confirmed_tracks.groupby("track_id")):
            # 绘制每条确认航迹，用不同符号区分
            track_table = track_group.sort_values("frame_idx")
            marker = TRACK_MARKER_LIST[track_order % len(TRACK_MARKER_LIST)]
            axis.plot(
                track_table["center_x"],
                track_table["center_y"],
                color=NEUTRAL_BLACK,
                linewidth=GLOBAL_TRACK_LINE_WIDTH + 0.35,
                marker=marker,
                markersize=5.5,
                zorder=8,
                label=f"T{int(track_id)}确认航迹",
            )
    axis.set_title("候选簇空间分布与确认航迹", pad=5)
    set_equal_axis(axis)
    axis.legend(loc="upper left", frameon=False, handletextpad=0.45)
    save_figure(figure, "图4_候选簇与确认航迹.png")


def set_track_view(axis: plt.Axes, track_table: pd.DataFrame) -> None:
    # 设置确认航迹图展示范围
    x_min = track_table["center_x"].min() - TRACK_VIEW_PADDING_X_KM
    x_max = track_table["center_x"].max() + TRACK_VIEW_PADDING_X_KM
    y_min = track_table["center_y"].min() - TRACK_VIEW_PADDING_Y_KM
    y_max = track_table["center_y"].max() + TRACK_VIEW_PADDING_Y_KM
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_xticks(np.linspace(x_min, x_max, TRACK_AXIS_TICK_COUNT))
    axis.set_yticks(np.linspace(y_min, y_max, TRACK_AXIS_TICK_COUNT))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def plot_confirmed_track(
    point_table: pd.DataFrame,
    confirmed_tracks: pd.DataFrame,
    frame_summary: pd.DataFrame,
) -> None:
    # 绘制全部确认航迹的局部放大图，每条按自身范围缩放，图4已含全局总览
    if confirmed_tracks.empty:
        return

    track_groups = list(confirmed_tracks.sort_values(["track_id", "frame_idx"]).groupby("track_id"))
    track_count = len(track_groups)
    figure, axes = plt.subplots(
        nrows=1,
        ncols=track_count,
        figsize=(4.9 * track_count, 4.2),
        constrained_layout=True,
    )
    flat_axes = [axes] if track_count == 1 else axes
    for track_order, ((track_id, track_group), axis) in enumerate(zip(track_groups, flat_axes)):
        track_table = track_group.sort_values("frame_idx").copy()
        track_table["frame_idx"] = track_table["frame_idx"].astype(int)
        track_table = track_table.merge(frame_summary[["frame_idx", "frame_time"]], on="frame_idx", how="left")
        frame_ids = track_table["frame_idx"].tolist()
        background_table = point_table[point_table["frame_idx"].isin(frame_ids)]
        track_color = TRACK_COLOR_LIST[track_order % len(TRACK_COLOR_LIST)]

        axis.scatter(
            background_table["x"],
            background_table["y"],
            s=TRACK_BACKGROUND_POINT_SIZE,
            color=NEUTRAL_DARK,
            alpha=TRACK_BACKGROUND_ALPHA,
            linewidths=0,
            label="对应帧点迹",
        )
        axis.plot(
            track_table["center_x"],
            track_table["center_y"],
            color=track_color,
            linewidth=TRACK_LINE_WIDTH,
            marker="o",
            markersize=4.2,
            label=f"T{int(track_id)}航迹中心",
        )
        for start_row, end_row in zip(track_table.itertuples(), track_table.iloc[1:].itertuples()):
            axis.annotate(
                "",
                xy=(end_row.center_x, end_row.center_y),
                xytext=(start_row.center_x, start_row.center_y),
                arrowprops={"arrowstyle": "->", "color": track_color, "lw": 0.8, "shrinkA": 5, "shrinkB": 5},
            )
        axis.scatter(
            [track_table["center_x"].iloc[0]],
            [track_table["center_y"].iloc[0]],
            s=TRACK_POINT_SIZE,
            color=RED_STRONG,
            linewidths=0,
            label="起点",
            zorder=6,
        )
        axis.scatter(
            [track_table["center_x"].iloc[-1]],
            [track_table["center_y"].iloc[-1]],
            s=TRACK_POINT_SIZE,
            color=NEUTRAL_BLACK,
            linewidths=0,
            label="终点",
            zorder=6,
        )
        start_time = str(track_table["frame_time"].iloc[0]).split()[-1]
        end_time = str(track_table["frame_time"].iloc[-1]).split()[-1]
        axis.set_title(
            f"T{int(track_id)}（F{frame_ids[0]}-F{frame_ids[-1]}，{start_time}-{end_time}，{len(track_table)}帧）",
            pad=5,
        )
        axis.set_xlabel("x / km")
        axis.set_ylabel("y / km")
        axis.grid(True, alpha=0.38)
        set_track_view(axis, track_table)
        axis.legend(loc="upper left", frameon=False, handletextpad=0.45)
    save_figure(figure, "图5_确认航迹局部放大.png")


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
    save_figure(figure, "图4_单帧候选检测.png")


def summarize_track_quality(track_table: pd.DataFrame) -> pd.DataFrame:
    # 汇总候选航迹质量指标
    if track_table.empty:
        return pd.DataFrame()
    ordered_tracks = track_table.sort_values(["track_id", "frame_idx"])
    quality_rows = []
    for track_id, track_group in ordered_tracks.groupby("track_id"):
        # 计算单条候选航迹质量
        coordinate_values = track_group[["center_x", "center_y"]].to_numpy(dtype=float)
        step_distances = np.linalg.norm(np.diff(coordinate_values, axis=0), axis=1)
        path_length = float(step_distances.sum())
        displacement = float(np.linalg.norm(coordinate_values[-1] - coordinate_values[0]))
        straightness = displacement / path_length if path_length > 0 else 0.0
        turn_angles = calculate_track_turn_angles(coordinate_values)
        quality_rows.append(
            {
                "track_id": int(track_id),
                "frame_count": int(len(track_group)),
                "mean_snr": float(track_group["mean_snr"].mean()),
                "straightness": straightness,
                "mean_step": float(step_distances.mean()) if len(step_distances) else 0.0,
                "max_step": float(step_distances.max()) if len(step_distances) else 0.0,
                "max_turn_angle": float(max(turn_angles)) if turn_angles else 0.0,
                "turn_angle_count": int(len(turn_angles)),
            }
        )
    return pd.DataFrame(quality_rows)


def calculate_track_turn_angles(coordinate_values: np.ndarray) -> list[float]:
    # 计算候选航迹的有效转向角
    if len(coordinate_values) < 3:
        return []

    step_vectors = np.diff(coordinate_values, axis=0)
    turn_angles = []
    for previous_step, current_step in zip(step_vectors[:-1], step_vectors[1:]):
        previous_distance = float(np.linalg.norm(previous_step))
        current_distance = float(np.linalg.norm(current_step))
        if previous_distance < MIN_DIRECTION_STEP_KM or current_distance < MIN_DIRECTION_STEP_KM:
            continue
        cosine_value = float(np.dot(previous_step, current_step) / (previous_distance * current_distance))
        clipped_cosine = float(np.clip(cosine_value, -1.0, 1.0))
        turn_angles.append(float(np.degrees(np.arccos(clipped_cosine))))
    return turn_angles


def plot_filter_funnel(
    cluster_table: pd.DataFrame,
    track_table: pd.DataFrame,
    confirmed_tracks: pd.DataFrame,
) -> None:
    # 绘制候选筛选数量评价图
    track_quality = summarize_track_quality(track_table)
    length_track_count = int((track_quality["frame_count"] >= MIN_TRACK_LENGTH).sum()) if not track_quality.empty else 0
    confirmed_track_count = confirmed_tracks["track_id"].nunique() if not confirmed_tracks.empty else 0
    metric_names = ["候选簇", "候选航迹", "长度达标", "质量确认"]
    metric_values = [
        int(len(cluster_table)),
        int(track_table["track_id"].nunique()) if not track_table.empty else 0,
        length_track_count,
        int(confirmed_track_count),
    ]

    figure, axis = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    bar_colors = [NEUTRAL_MID, BLUE_SECONDARY, BLUE_MAIN, RED_STRONG]
    axis.bar(metric_names, metric_values, width=METRIC_BAR_WIDTH, color=bar_colors)
    axis.set_title("候选筛选数量评价", pad=5)
    axis.set_ylabel("数量")
    axis.grid(axis="y", alpha=0.28)
    for metric_index, metric_value in enumerate(metric_values):
        # 标注每级筛选数量
        label_y = metric_value + max(metric_values) * METRIC_LABEL_PADDING
        axis.text(metric_index, label_y, str(metric_value), ha="center")
    save_figure(figure, "图6_候选筛选数量评价.png")


def plot_track_quality(track_table: pd.DataFrame) -> None:
    # 绘制航迹质量指标评价图
    track_quality = summarize_track_quality(track_table)
    if track_quality.empty:
        return
    display_quality = track_quality[track_quality["frame_count"] >= MIN_TRACK_LENGTH].copy()
    if display_quality.empty:
        return
    display_quality = display_quality.sort_values("track_id")
    track_labels = [f"T{int(track_id)}" for track_id in display_quality["track_id"]]

    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(9.2, 3.25), constrained_layout=True)
    axes[0].bar(track_labels, display_quality["straightness"], color=BLUE_MAIN, width=METRIC_BAR_WIDTH)
    axes[0].axhline(MIN_TRACK_STRAIGHTNESS, color=RED_STRONG, linestyle="--", linewidth=0.9, label="确认阈值")
    axes[0].set_title("航迹直线性", pad=5)
    axes[0].set_ylabel("straightness")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis="y", alpha=0.28)
    axes[0].legend(frameon=False)

    track_positions = np.arange(len(track_labels))
    step_bar_width = METRIC_BAR_WIDTH / 2
    axes[1].bar(
        track_positions - step_bar_width / 2,
        display_quality["max_step"],
        color=BLUE_SECONDARY,
        width=step_bar_width,
        label="最大跳变",
    )
    axes[1].bar(
        track_positions + step_bar_width / 2,
        display_quality["mean_step"],
        color=NEUTRAL_MID,
        width=step_bar_width,
        label="平均步长",
    )
    axes[1].axhline(MAX_TRACK_STEP_KM, color=RED_STRONG, linestyle="--", linewidth=0.9, label="最大步长阈值")
    axes[1].axhline(MAX_MEAN_TRACK_STEP_KM, color=NEUTRAL_BLACK, linestyle=":", linewidth=0.9, label="平均步长阈值")
    axes[1].set_title("相邻帧步长", pad=5)
    axes[1].set_ylabel("step / km")
    axes[1].set_xticks(track_positions)
    axes[1].set_xticklabels(track_labels)
    axes[1].grid(axis="y", alpha=0.28)
    axes[1].legend(frameon=False)

    axes[2].bar(track_labels, display_quality["max_turn_angle"], color=BLUE_MAIN, width=METRIC_BAR_WIDTH)
    axes[2].axhline(MAX_DIRECTION_CHANGE_DEG, color=RED_STRONG, linestyle="--", linewidth=0.9, label="转向阈值")
    axes[2].set_title("最大转向角", pad=5)
    axes[2].set_ylabel("angle / deg")
    axes[2].grid(axis="y", alpha=0.28)
    for track_index, turn_angle_count in enumerate(display_quality["turn_angle_count"]):
        # 标注没有有效转向角的航迹
        if int(turn_angle_count) == 0:
            axes[2].text(
                track_index,
                TURN_NOTE_Y_DEG,
                TURN_NOTE_TEXT,
                ha="center",
                va="bottom",
                color=NEUTRAL_DARK,
                fontsize=6.8,
                rotation=90,
            )
    axes[2].legend(frameon=False)
    save_figure(figure, "图7_航迹质量评价.png")


def main():
    # 设置控制台编码和绘图风格
    sys.stdout.reconfigure(encoding="utf-8")
    ensure_output_dirs()
    configure_plot_style()

    # 读取数据和检测结果
    point_table = load_point_table()
    strong_point_table = pd.read_csv(TABLE_DIR / "strong_points.csv")
    cluster_table = read_result_table("candidate_clusters.csv")
    track_table = read_result_table("candidate_tracks.csv")
    confirmed_tracks = read_result_table("confirmed_tracks.csv")
    frame_summary = pd.read_csv(TABLE_DIR / "frame_summary.csv")

    # 生成报告主图和过程图
    plot_spatial_density(point_table)
    plot_spatial_distribution(point_table)
    plot_multi_frame_detection(point_table, strong_point_table, cluster_table, confirmed_tracks)
    plot_global_candidate_clusters(point_table, cluster_table, confirmed_tracks)
    plot_confirmed_track(point_table, confirmed_tracks, frame_summary)
    plot_filter_funnel(cluster_table, track_table, confirmed_tracks)
    plot_track_quality(track_table)

    # 输出图片目录
    print(f"主图已保存：{FIGURE_DIR}")


if __name__ == "__main__":
    # 执行报告主图生成流程
    main()
