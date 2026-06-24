import sys

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from hfr_config import (
    AMP_QUANTILE,
    AMP_SCALE,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    LOCAL_RANGE_BIN_COUNT,
    LOCAL_RANGE_MIN_POINTS,
    MAX_DIRECTION_CHANGE_DEG,
    MAX_MEAN_TRACK_STEP_KM,
    MAX_LINK_DISTANCE_KM,
    MAX_TRACK_GAP_FRAMES,
    MAX_TRACK_STEP_KM,
    MAX_VECTOR_VELOCITY_DIFF_KMH,
    MAX_VELOCITY_DIFF_KMH,
    MIN_DIRECTION_STEP_KM,
    MIN_CLUSTER_SIZE,
    MIN_TRACK_LENGTH,
    MIN_TRACK_STRAIGHTNESS,
    RESULT_DIR,
    SECONDS_PER_HOUR,
    SIGNAL_SCORE_MIN,
    SNR_QUANTILE,
    SNR_SCALE_DB,
    SPACE_SCALE_KM,
    TABLE_DIR,
    TRACK_FRAME_INTERVAL_SECONDS,
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
# 确认航迹字段来源：空结果也保留稳定表头
TRACK_RESULT_COLUMNS = [
    "frame_idx",
    "cluster_id",
    "point_count",
    "center_x",
    "center_y",
    "center_lon",
    "center_lat",
    "mean_velocity",
    "mean_vx",
    "mean_vy",
    "mean_snr",
    "mean_amp",
    "score",
    "track_id",
]
# 航迹摘要字段来源：报告分析表格固定读取
TRACK_SUMMARY_COLUMNS = [
    "track_id",
    "frame_count",
    "start_frame",
    "end_frame",
    "mean_snr",
    "mean_velocity",
    "mean_vx",
    "mean_vy",
    "mean_point_count",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "displacement",
    "path_length",
    "straightness",
    "mean_step",
    "max_step",
    "max_turn_angle",
]
# 航迹允许跨帧来源：断帧数加当前关联帧
MAX_TRACK_FRAME_STEP = MAX_TRACK_GAP_FRAMES + 1
# 加权中心基准来源：保证权重为正
CLUSTER_WEIGHT_EPS = 1e-6


def calculate_signal_weights(cluster_table: pd.DataFrame) -> np.ndarray:
    # 计算候选簇内点迹信号权重
    snr_values = cluster_table["snr"].to_numpy(dtype=float)
    amp_values = cluster_table["amp"].to_numpy(dtype=float)
    snr_weights = snr_values - np.nanmin(snr_values) + CLUSTER_WEIGHT_EPS
    amp_weights = amp_values - np.nanmin(amp_values) + CLUSTER_WEIGHT_EPS
    signal_weights = snr_weights + amp_weights
    if not np.isfinite(signal_weights).all() or float(signal_weights.sum()) <= 0:
        return np.ones(len(cluster_table), dtype=float)
    return signal_weights


def weighted_mean(cluster_table: pd.DataFrame, column_name: str, weights: np.ndarray) -> float:
    # 计算指定字段加权均值
    column_values = cluster_table[column_name].to_numpy(dtype=float)
    return float(np.average(column_values, weights=weights))


def select_strong_points(frame_table: pd.DataFrame) -> pd.DataFrame:
    # 按帧和距离分区执行信号强点筛选
    valid_table = frame_table.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["snr", "amp", "x", "y", "velocity", "range"]
    )
    if valid_table.empty:
        return valid_table

    valid_table = valid_table.copy()
    frame_snr_threshold = valid_table["snr"].quantile(SNR_QUANTILE)
    frame_amp_threshold = valid_table["amp"].quantile(AMP_QUANTILE)
    valid_table["local_snr_threshold"] = frame_snr_threshold
    valid_table["local_amp_threshold"] = frame_amp_threshold

    range_bin_count = min(LOCAL_RANGE_BIN_COUNT, max(1, len(valid_table) // LOCAL_RANGE_MIN_POINTS))
    if range_bin_count > 1:
        range_bins = pd.qcut(valid_table["range"], q=range_bin_count, duplicates="drop")
        for _, range_group in valid_table.groupby(range_bins, observed=False):
            # 更新距离分区阈值
            if len(range_group) < LOCAL_RANGE_MIN_POINTS:
                continue
            local_snr_threshold = range_group["snr"].quantile(SNR_QUANTILE)
            local_amp_threshold = range_group["amp"].quantile(AMP_QUANTILE)
            valid_table.loc[range_group.index, "local_snr_threshold"] = min(
                frame_snr_threshold,
                local_snr_threshold,
            )
            valid_table.loc[range_group.index, "local_amp_threshold"] = min(
                frame_amp_threshold,
                local_amp_threshold,
            )

    valid_table["snr_rank"] = valid_table["snr"].rank(pct=True)
    valid_table["amp_rank"] = valid_table["amp"].rank(pct=True)
    valid_table["signal_score"] = valid_table["snr_rank"] + valid_table["amp_rank"]

    local_signal_mask = (
        (valid_table["snr"] >= valid_table["local_snr_threshold"])
        & (valid_table["amp"] >= valid_table["local_amp_threshold"])
    )
    combined_signal_mask = valid_table["signal_score"] >= SIGNAL_SCORE_MIN
    strong_table = valid_table[local_signal_mask | combined_signal_mask].copy()
    return strong_table


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

    cluster_features = build_cluster_features(strong_points)
    cluster_labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(cluster_features)
    strong_points["cluster_id"] = cluster_labels
    cluster_rows = []

    for cluster_id in sorted(label for label in np.unique(cluster_labels) if label >= 0):
        cluster_table = strong_points[strong_points["cluster_id"] == cluster_id]
        if len(cluster_table) < MIN_CLUSTER_SIZE:
            continue
        signal_weights = calculate_signal_weights(cluster_table)
        cluster_rows.append(
            {
                "frame_idx": int(cluster_table["frame_idx"].iloc[0]),
                "cluster_id": int(cluster_id),
                "point_count": int(len(cluster_table)),
                "center_x": weighted_mean(cluster_table, "x", signal_weights),
                "center_y": weighted_mean(cluster_table, "y", signal_weights),
                "center_lon": weighted_mean(cluster_table, "lon", signal_weights),
                "center_lat": weighted_mean(cluster_table, "lat", signal_weights),
                "mean_velocity": weighted_mean(cluster_table, "velocity", signal_weights),
                "mean_vx": weighted_mean(cluster_table, "vx", signal_weights),
                "mean_vy": weighted_mean(cluster_table, "vy", signal_weights),
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
    # 计算候选簇与航迹预测位置距离
    frame_step = int(candidate_row["frame_idx"] - track_state["last_frame"])
    elapsed_hours = frame_step * TRACK_FRAME_INTERVAL_SECONDS / SECONDS_PER_HOUR
    predicted_x = float(track_state["last_x"] + track_state["last_vx"] * elapsed_hours)
    predicted_y = float(track_state["last_y"] + track_state["last_vy"] * elapsed_hours)
    delta_x = float(candidate_row["center_x"] - predicted_x)
    delta_y = float(candidate_row["center_y"] - predicted_y)
    return float(np.hypot(delta_x, delta_y))


def calculate_direction_change(candidate_row: pd.Series, track_state: dict) -> float:
    # 计算候选延续方向与上一段方向的夹角
    if track_state.get("previous_x") is None:
        return 0.0

    previous_step = np.array(
        [
            track_state["last_x"] - track_state["previous_x"],
            track_state["last_y"] - track_state["previous_y"],
        ],
        dtype=float,
    )
    current_step = np.array(
        [
            candidate_row["center_x"] - track_state["last_x"],
            candidate_row["center_y"] - track_state["last_y"],
        ],
        dtype=float,
    )
    previous_distance = float(np.linalg.norm(previous_step))
    current_distance = float(np.linalg.norm(current_step))
    if previous_distance < MIN_DIRECTION_STEP_KM or current_distance < MIN_DIRECTION_STEP_KM:
        return 0.0

    cosine_value = float(np.dot(previous_step, current_step) / (previous_distance * current_distance))
    clipped_cosine = float(np.clip(cosine_value, -1.0, 1.0))
    return float(np.degrees(np.arccos(clipped_cosine)))


def is_direction_consistent(candidate_row: pd.Series, track_state: dict) -> bool:
    # 判断候选延续方向是否平滑
    turn_angle = calculate_direction_change(candidate_row, track_state)
    return turn_angle <= MAX_DIRECTION_CHANGE_DEG


def find_track_match(frame_candidates: pd.DataFrame, track_state: dict, used_indices: set) -> int | None:
    # 选择当前航迹的最近邻候选
    best_index = None
    best_distance = MAX_LINK_DISTANCE_KM
    for candidate_index, candidate_row in frame_candidates.iterrows():
        if candidate_index in used_indices:
            continue
        distance = candidate_distance(candidate_row, track_state)
        velocity_diff = abs(float(candidate_row["mean_velocity"] - track_state["last_velocity"]))
        vector_velocity_diff = np.hypot(
            float(candidate_row["mean_vx"] - track_state["last_vx"]),
            float(candidate_row["mean_vy"] - track_state["last_vy"]),
        )
        if (
            distance <= best_distance
            and velocity_diff <= MAX_VELOCITY_DIFF_KMH
            and vector_velocity_diff <= MAX_VECTOR_VELOCITY_DIFF_KMH
            and is_direction_consistent(candidate_row, track_state)
        ):
            best_index = candidate_index
            best_distance = distance
    return best_index


def link_tracks(cluster_table: pd.DataFrame) -> pd.DataFrame:
    # 使用预测位置和最近邻逻辑关联候选簇
    if cluster_table.empty:
        return pd.DataFrame()

    active_tracks = []
    track_rows = []
    next_track_id = 1

    for frame_idx in sorted(cluster_table["frame_idx"].unique()):
        frame_candidates = cluster_table[cluster_table["frame_idx"] == frame_idx].sort_values("score", ascending=False)
        used_indices = set()

        for track_state in active_tracks:
            frame_step = int(frame_idx - track_state["last_frame"])
            if frame_step < 1 or frame_step > MAX_TRACK_FRAME_STEP:
                continue
            best_index = find_track_match(frame_candidates, track_state, used_indices)
            if best_index is None:
                continue
            matched_row = frame_candidates.loc[best_index].copy()
            matched_row["track_id"] = track_state["track_id"]
            track_rows.append(matched_row)
            used_indices.add(best_index)
            track_state["previous_x"] = track_state["last_x"]
            track_state["previous_y"] = track_state["last_y"]
            track_state["last_frame"] = int(matched_row["frame_idx"])
            track_state["last_x"] = float(matched_row["center_x"])
            track_state["last_y"] = float(matched_row["center_y"])
            track_state["last_velocity"] = float(matched_row["mean_velocity"])
            track_state["last_vx"] = float(matched_row["mean_vx"])
            track_state["last_vy"] = float(matched_row["mean_vy"])

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
                    "previous_x": None,
                    "previous_y": None,
                    "last_x": float(candidate_row["center_x"]),
                    "last_y": float(candidate_row["center_y"]),
                    "last_velocity": float(candidate_row["mean_velocity"]),
                    "last_vx": float(candidate_row["mean_vx"]),
                    "last_vy": float(candidate_row["mean_vy"]),
                }
            )
            next_track_id += 1

        # 移除无法继续关联的历史航迹
        active_tracks = [
            track_state
            for track_state in active_tracks
            if frame_idx - track_state["last_frame"] <= MAX_TRACK_FRAME_STEP
        ]

    if not track_rows:
        return pd.DataFrame(columns=TRACK_RESULT_COLUMNS)
    return pd.DataFrame(track_rows)[TRACK_RESULT_COLUMNS]


def calculate_track_turn_angles(coordinate_values: np.ndarray) -> list[float]:
    # 计算整条航迹的有效转向角
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


def summarize_tracks(confirmed_tracks: pd.DataFrame) -> pd.DataFrame:
    # 汇总确认航迹的关键指标
    if confirmed_tracks.empty:
        return pd.DataFrame(columns=TRACK_SUMMARY_COLUMNS)
    ordered_tracks = confirmed_tracks.sort_values(["track_id", "frame_idx"])
    track_summary = (
        ordered_tracks.groupby("track_id")
        .agg(
            frame_count=("frame_idx", "count"),
            start_frame=("frame_idx", "min"),
            end_frame=("frame_idx", "max"),
            mean_snr=("mean_snr", "mean"),
            mean_velocity=("mean_velocity", "mean"),
            mean_vx=("mean_vx", "mean"),
            mean_vy=("mean_vy", "mean"),
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
        turn_angles = calculate_track_turn_angles(coordinate_values)
        quality_rows.append(
            {
                "track_id": track_id,
                "path_length": path_length,
                "straightness": straightness,
                "mean_step": float(step_distances.mean()) if len(step_distances) else 0.0,
                "max_step": float(step_distances.max()) if len(step_distances) else 0.0,
                "max_turn_angle": float(max(turn_angles)) if turn_angles else 0.0,
            }
        )
    quality_table = pd.DataFrame(quality_rows)
    track_summary = track_summary.merge(quality_table, on="track_id", how="left")
    sorted_summary = track_summary.sort_values(["straightness", "frame_count", "mean_snr"], ascending=False)
    return sorted_summary[TRACK_SUMMARY_COLUMNS]


def filter_track_summary(track_summary: pd.DataFrame) -> pd.DataFrame:
    # 筛选形态质量达标的航迹摘要
    if track_summary.empty:
        return track_summary
    quality_mask = (
        (track_summary["straightness"] >= MIN_TRACK_STRAIGHTNESS)
        & (track_summary["max_step"] <= MAX_TRACK_STEP_KM)
        & (track_summary["mean_step"] <= MAX_MEAN_TRACK_STEP_KM)
        & (track_summary["max_turn_angle"] <= MAX_DIRECTION_CHANGE_DEG)
    )
    return track_summary[quality_mask].copy()


def select_confirmed_tracks(track_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 按长度和形态质量确认疑似航迹
    if track_table.empty:
        empty_tracks = pd.DataFrame(columns=TRACK_RESULT_COLUMNS)
        empty_summary = pd.DataFrame(columns=TRACK_SUMMARY_COLUMNS)
        return empty_tracks, empty_summary, empty_summary

    track_lengths = track_table.groupby("track_id").size()
    length_track_ids = track_lengths[track_lengths >= MIN_TRACK_LENGTH].index
    length_tracks = track_table[track_table["track_id"].isin(length_track_ids)].copy()
    length_summary = summarize_tracks(length_tracks)
    quality_summary = filter_track_summary(length_summary)
    confirmed_track_ids = quality_summary["track_id"].astype(int).tolist()
    confirmed_tracks = track_table[track_table["track_id"].isin(confirmed_track_ids)].copy()
    track_summary = summarize_tracks(confirmed_tracks)
    return confirmed_tracks, track_summary, length_summary


def main():
    # 设置控制台编码和输出目录
    sys.stdout.reconfigure(encoding="utf-8")
    ensure_output_dirs()

    # 读取点迹并执行检测跟踪
    point_table = load_point_table()
    strong_point_table, cluster_table = detect_all_clusters(point_table)
    track_table = link_tracks(cluster_table)
    confirmed_tracks, track_summary, length_summary = select_confirmed_tracks(track_table)

    # 保存结果数据
    strong_point_table.to_csv(TABLE_DIR / "strong_points.csv", index=False, encoding="utf-8-sig")
    cluster_table.to_csv(RESULT_DIR / CLUSTER_FILE_NAME, index=False, encoding="utf-8-sig")
    track_table.to_csv(RESULT_DIR / TRACK_FILE_NAME, index=False, encoding="utf-8-sig")
    confirmed_tracks.to_csv(RESULT_DIR / CONFIRMED_TRACK_FILE_NAME, index=False, encoding="utf-8-sig")
    track_summary.to_csv(RESULT_DIR / TRACK_SUMMARY_FILE_NAME, index=False, encoding="utf-8-sig")

    # 输出运行摘要
    print(f"候选簇数量：{len(cluster_table)}")
    print(f"候选航迹数量：{track_table['track_id'].nunique() if not track_table.empty else 0}")
    print(f"长度达标航迹数量：{length_summary['track_id'].nunique() if not length_summary.empty else 0}")
    print(f"质量确认航迹数量：{confirmed_tracks['track_id'].nunique() if not confirmed_tracks.empty else 0}")


if __name__ == "__main__":
    # 执行检测与航迹确认流程
    main()
