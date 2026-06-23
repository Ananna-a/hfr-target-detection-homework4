import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hfr_detection import detect_all_frames
from hfr_utils import (
    RESULT_DIR,
    all_frames_to_dataframe,
    configure_chinese_plot,
    ensure_output_dirs,
    load_hfr_data,
    save_figure,
)


# 最大关联距离来源：相邻帧候选中心允许的空间偏差
MAX_LINK_DISTANCE_KM = 12.0
# 最大速度差来源：船舶目标相邻帧速度应保持连续
MAX_VELOCITY_DIFF_KMH = 25.0
# 最小航迹长度来源：过滤短寿命虚警候选
MIN_TRACK_LENGTH = 3
# 图像宽度来源：报告轨迹图尺寸
FIGURE_WIDTH = 10
# 图像高度来源：报告轨迹图尺寸
FIGURE_HEIGHT = 8
# 航迹结果文件名来源：保存全部候选关联结果
TRACK_RESULT_FILE_NAME = "候选目标航迹.csv"
# 确认航迹文件名来源：保存满足长度要求的航迹
CONFIRMED_TRACK_FILE_NAME = "确认目标航迹.csv"


def calculate_distance(row_data: pd.Series, track_state: dict) -> float:
    # 计算候选点与航迹末端的平面距离
    delta_x = float(row_data["center_x"] - track_state["last_x"])
    delta_y = float(row_data["center_y"] - track_state["last_y"])
    return float(np.sqrt(delta_x * delta_x + delta_y * delta_y))


def find_best_match(frame_candidates: pd.DataFrame, track_state: dict, used_indices: set) -> int | None:
    # 寻找当前航迹在本帧的最佳候选匹配
    best_index = None
    best_distance = MAX_LINK_DISTANCE_KM
    for candidate_index, candidate_row in frame_candidates.iterrows():
        if candidate_index in used_indices:
            continue
        distance = calculate_distance(candidate_row, track_state)
        velocity_diff = abs(float(candidate_row["mean_velocity"] - track_state["last_velocity"]))
        if distance <= best_distance and velocity_diff <= MAX_VELOCITY_DIFF_KMH:
            best_index = candidate_index
            best_distance = distance
    return best_index


def link_candidate_clusters(candidate_clusters: pd.DataFrame) -> pd.DataFrame:
    # 将相邻帧目标候选簇关联为简化航迹
    if candidate_clusters.empty:
        return pd.DataFrame()

    next_track_id = 1
    active_tracks = []
    track_rows = []

    for frame_id in sorted(candidate_clusters["frame_idx"].unique()):
        frame_candidates = candidate_clusters[candidate_clusters["frame_idx"] == frame_id].sort_values("score", ascending=False)
        used_indices = set()

        for track_state in active_tracks:
            if frame_id - track_state["last_frame"] != 1:
                continue
            best_index = find_best_match(frame_candidates, track_state, used_indices)
            if best_index is None:
                continue
            used_indices.add(best_index)
            matched_row = frame_candidates.loc[best_index].copy()
            matched_row["track_id"] = track_state["track_id"]
            track_rows.append(matched_row)
            track_state["last_frame"] = int(matched_row["frame_idx"])
            track_state["last_x"] = float(matched_row["center_x"])
            track_state["last_y"] = float(matched_row["center_y"])
            track_state["last_velocity"] = float(matched_row["mean_velocity"])

        for candidate_index, candidate_row in frame_candidates.iterrows():
            if candidate_index in used_indices:
                continue
            new_row = candidate_row.copy()
            new_row["track_id"] = next_track_id
            track_rows.append(new_row)
            active_tracks.append(
                {
                    "track_id": next_track_id,
                    "last_frame": int(candidate_row["frame_idx"]),
                    "last_x": float(candidate_row["center_x"]),
                    "last_y": float(candidate_row["center_y"]),
                    "last_velocity": float(candidate_row["mean_velocity"]),
                }
            )
            next_track_id += 1

    return pd.DataFrame(track_rows)


def filter_confirmed_tracks(track_table: pd.DataFrame) -> pd.DataFrame:
    # 保留连续帧数达到要求的航迹
    if track_table.empty:
        return pd.DataFrame()
    track_lengths = track_table.groupby("track_id").size()
    valid_track_ids = track_lengths[track_lengths >= MIN_TRACK_LENGTH].index
    return track_table[track_table["track_id"].isin(valid_track_ids)].copy()


def plot_track_result(candidate_clusters: pd.DataFrame, confirmed_tracks: pd.DataFrame):
    # 绘制确认航迹结果图
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    if not candidate_clusters.empty:
        axis.scatter(
            candidate_clusters["center_x"],
            candidate_clusters["center_y"],
            s=24,
            color="#bdbdbd",
            alpha=0.55,
            label="候选目标中心",
        )

    if not confirmed_tracks.empty:
        for track_id, track_table in confirmed_tracks.groupby("track_id"):
            ordered_track = track_table.sort_values("frame_idx")
            axis.plot(
                ordered_track["center_x"],
                ordered_track["center_y"],
                marker="o",
                linewidth=1.8,
                label=f"航迹{int(track_id)}",
            )

    axis.set_title("多帧确认目标航迹")
    axis.set_xlabel("x坐标/km")
    axis.set_ylabel("y坐标/km")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(loc="best", fontsize=8)
    axis.set_aspect("equal", adjustable="box")
    return save_figure(figure, "多帧确认目标航迹.png")


def main():
    # 设置控制台编码和中文绘图
    sys.stdout.reconfigure(encoding="utf-8")
    configure_chinese_plot()
    ensure_output_dirs()

    # 检测候选簇并执行航迹关联
    hfr_data = load_hfr_data()
    point_table = all_frames_to_dataframe(hfr_data)
    _, candidate_clusters = detect_all_frames(point_table)
    track_table = link_candidate_clusters(candidate_clusters)
    confirmed_tracks = filter_confirmed_tracks(track_table)

    # 保存航迹结果
    track_result_path = RESULT_DIR / TRACK_RESULT_FILE_NAME
    confirmed_track_path = RESULT_DIR / CONFIRMED_TRACK_FILE_NAME
    track_table.to_csv(track_result_path, index=False, encoding="utf-8-sig")
    confirmed_tracks.to_csv(confirmed_track_path, index=False, encoding="utf-8-sig")
    figure_path = plot_track_result(candidate_clusters, confirmed_tracks)

    # 输出航迹摘要
    print(f"候选目标航迹已保存：{track_result_path}")
    print(f"确认目标航迹已保存：{confirmed_track_path}")
    print(f"航迹结果图已保存：{figure_path}")
    print(f"候选航迹数量：{track_table['track_id'].nunique() if not track_table.empty else 0}")
    print(f"确认航迹数量：{confirmed_tracks['track_id'].nunique() if not confirmed_tracks.empty else 0}")


if __name__ == "__main__":
    # 执行多帧目标确认流程
    main()
