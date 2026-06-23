import sys

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from hfr_config import (
    AMP_QUANTILE,
    AMP_SCALE,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    MAX_LINK_DISTANCE_KM,
    MAX_VELOCITY_DIFF_KMH,
    MIN_CLUSTER_SIZE,
    MIN_TRACK_LENGTH,
    RESULT_DIR,
    SNR_QUANTILE,
    SNR_SCALE_DB,
    SPACE_SCALE_KM,
    TABLE_DIR,
    VELOCITY_SCALE_KMH,
    ensure_output_dirs,
    load_point_table,
)


# 候选簇文件名来源：保存每帧检测出的目标候选
CLUSTER_FILE_NAME = "candidate_clusters.csv"
# 候选航迹文件名来源：保存所有关联链路
TRACK_FILE_NAME = "candidate_tracks.csv"
# 确认航迹文件名来源：保存稳定疑似目标航迹
CONFIRMED_TRACK_FILE_NAME = "confirmed_tracks.csv"
# 航迹摘要文件名来源：报告分析引用
TRACK_SUMMARY_FILE_NAME = "track_summary.csv"


def select_strong_points(frame_table: pd.DataFrame) -> pd.DataFrame:
    # 按帧执行信噪比和幅度自适应筛选
    valid_table = frame_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["snr", "amp", "x", "y", "velocity"])
    if valid_table.empty:
        return valid_table
    snr_threshold = valid_table["snr"].quantile(SNR_QUANTILE)
    amp_threshold = valid_table["amp"].quantile(AMP_QUANTILE)
    return valid_table[(valid_table["snr"] >= snr_threshold) & (valid_table["amp"] >= amp_threshold)].copy()


def build_cluster_features(point_table: pd.DataFrame) -> np.ndarray:
    # 将点迹特征转换到可比较的物理尺度
    return np.column_stack(
        [
            point_table["x"].to_numpy() / SPACE_SCALE_KM,
            point_table["y"].to_numpy() / SPACE_SCALE_KM,
            point_table["velocity"].to_numpy() / VELOCITY_SCALE_KMH,
            point_table["snr"].to_numpy() / SNR_SCALE_DB,
            point_table["amp"].to_numpy() / AMP_SCALE,
        ]
    )


def detect_frame_clusters(frame_table: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    # 对单帧强点进行聚类并汇总候选簇
    strong_points = select_strong_points(frame_table)
    if len(strong_points) < DBSCAN_MIN_SAMPLES:
        return strong_points.assign(cluster_id=-1), []

    cluster_labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(build_cluster_features(strong_points))
    strong_points["cluster_id"] = cluster_labels
    cluster_rows = []

    for cluster_id in sorted(label for label in np.unique(cluster_labels) if label >= 0):
        cluster_table = strong_points[strong_points["cluster_id"] == cluster_id]
        if len(cluster_table) < MIN_CLUSTER_SIZE:
            continue
        cluster_rows.append(
            {
                "frame_idx": int(cluster_table["frame_idx"].iloc[0]),
                "cluster_id": int(cluster_id),
                "point_count": int(len(cluster_table)),
                "center_x": float(cluster_table["x"].mean()),
                "center_y": float(cluster_table["y"].mean()),
                "center_lon": float(cluster_table["lon"].mean()),
                "center_lat": float(cluster_table["lat"].mean()),
                "mean_velocity": float(cluster_table["velocity"].mean()),
                "mean_snr": float(cluster_table["snr"].mean()),
                "mean_amp": float(cluster_table["amp"].mean()),
                "score": float(cluster_table["snr"].mean() + np.log1p(len(cluster_table))),
            }
        )
    return strong_points, cluster_rows


def detect_all_clusters(point_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 对全部帧执行候选检测
    strong_point_tables = []
    cluster_rows = []
    for _, frame_table in point_table.groupby("frame_idx"):
        strong_points, frame_cluster_rows = detect_frame_clusters(frame_table)
        strong_point_tables.append(strong_points)
        cluster_rows.extend(frame_cluster_rows)
    strong_point_table = pd.concat(strong_point_tables, ignore_index=True)
    cluster_table = pd.DataFrame(cluster_rows)
    return strong_point_table, cluster_table


def candidate_distance(candidate_row: pd.Series, track_state: dict) -> float:
    # 计算候选簇与航迹末端距离
    delta_x = float(candidate_row["center_x"] - track_state["last_x"])
    delta_y = float(candidate_row["center_y"] - track_state["last_y"])
    return float(np.hypot(delta_x, delta_y))


def find_track_match(frame_candidates: pd.DataFrame, track_state: dict, used_indices: set) -> int | None:
    # 选择当前航迹的最近邻候选
    best_index = None
    best_distance = MAX_LINK_DISTANCE_KM
    for candidate_index, candidate_row in frame_candidates.iterrows():
        if candidate_index in used_indices:
            continue
        distance = candidate_distance(candidate_row, track_state)
        velocity_diff = abs(float(candidate_row["mean_velocity"] - track_state["last_velocity"]))
        if distance <= best_distance and velocity_diff <= MAX_VELOCITY_DIFF_KMH:
            best_index = candidate_index
            best_distance = distance
    return best_index


def link_tracks(cluster_table: pd.DataFrame) -> pd.DataFrame:
    # 使用最近邻逻辑关联相邻帧候选簇
    if cluster_table.empty:
        return pd.DataFrame()

    active_tracks = []
    track_rows = []
    next_track_id = 1

    for frame_idx in sorted(cluster_table["frame_idx"].unique()):
        frame_candidates = cluster_table[cluster_table["frame_idx"] == frame_idx].sort_values("score", ascending=False)
        used_indices = set()

        for track_state in active_tracks:
            if frame_idx - track_state["last_frame"] != 1:
                continue
            best_index = find_track_match(frame_candidates, track_state, used_indices)
            if best_index is None:
                continue
            matched_row = frame_candidates.loc[best_index].copy()
            matched_row["track_id"] = track_state["track_id"]
            track_rows.append(matched_row)
            used_indices.add(best_index)
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


def summarize_tracks(confirmed_tracks: pd.DataFrame) -> pd.DataFrame:
    # 汇总确认航迹的关键指标
    if confirmed_tracks.empty:
        return pd.DataFrame()
    ordered_tracks = confirmed_tracks.sort_values(["track_id", "frame_idx"])
    track_summary = (
        ordered_tracks.groupby("track_id")
        .agg(
            frame_count=("frame_idx", "count"),
            start_frame=("frame_idx", "min"),
            end_frame=("frame_idx", "max"),
            mean_snr=("mean_snr", "mean"),
            mean_velocity=("mean_velocity", "mean"),
            mean_point_count=("point_count", "mean"),
            start_x=("center_x", "first"),
            start_y=("center_y", "first"),
            end_x=("center_x", "last"),
            end_y=("center_y", "last"),
        )
        .reset_index()
    )
    track_summary["displacement"] = np.hypot(
        track_summary["end_x"] - track_summary["start_x"],
        track_summary["end_y"] - track_summary["start_y"],
    )
    quality_rows = []
    for track_id, track_group in ordered_tracks.groupby("track_id"):
        # 计算航迹形态质量指标
        coordinate_values = track_group[["center_x", "center_y"]].to_numpy(dtype=float)
        step_distances = np.linalg.norm(np.diff(coordinate_values, axis=0), axis=1)
        path_length = float(step_distances.sum())
        displacement = float(np.linalg.norm(coordinate_values[-1] - coordinate_values[0]))
        straightness = displacement / path_length if path_length > 0 else 0.0
        quality_rows.append(
            {
                "track_id": track_id,
                "path_length": path_length,
                "straightness": straightness,
                "mean_step": float(step_distances.mean()) if len(step_distances) else 0.0,
                "max_step": float(step_distances.max()) if len(step_distances) else 0.0,
            }
        )
    quality_table = pd.DataFrame(quality_rows)
    track_summary = track_summary.merge(quality_table, on="track_id", how="left")
    return track_summary.sort_values(["straightness", "frame_count", "mean_snr"], ascending=False)


def main():
    # 设置控制台编码和输出目录
    sys.stdout.reconfigure(encoding="utf-8")
    ensure_output_dirs()

    # 读取点迹并执行检测跟踪
    point_table = load_point_table()
    strong_point_table, cluster_table = detect_all_clusters(point_table)
    track_table = link_tracks(cluster_table)
    track_lengths = track_table.groupby("track_id").size() if not track_table.empty else pd.Series(dtype=int)
    confirmed_track_ids = track_lengths[track_lengths >= MIN_TRACK_LENGTH].index
    confirmed_tracks = track_table[track_table["track_id"].isin(confirmed_track_ids)].copy()
    track_summary = summarize_tracks(confirmed_tracks)

    # 保存结果数据
    strong_point_table.to_csv(TABLE_DIR / "strong_points.csv", index=False, encoding="utf-8-sig")
    cluster_table.to_csv(RESULT_DIR / CLUSTER_FILE_NAME, index=False, encoding="utf-8-sig")
    track_table.to_csv(RESULT_DIR / TRACK_FILE_NAME, index=False, encoding="utf-8-sig")
    confirmed_tracks.to_csv(RESULT_DIR / CONFIRMED_TRACK_FILE_NAME, index=False, encoding="utf-8-sig")
    track_summary.to_csv(RESULT_DIR / TRACK_SUMMARY_FILE_NAME, index=False, encoding="utf-8-sig")

    # 输出运行摘要
    print(f"候选簇数量：{len(cluster_table)}")
    print(f"候选航迹数量：{track_table['track_id'].nunique() if not track_table.empty else 0}")
    print(f"确认航迹数量：{confirmed_tracks['track_id'].nunique() if not confirmed_tracks.empty else 0}")


if __name__ == "__main__":
    # 执行检测与航迹确认流程
    main()
