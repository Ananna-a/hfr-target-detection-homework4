import sys

import numpy as np
import pandas as pd

from hfr_config import (
    RESULT_DIR,
    TABLE_DIR,
    MIN_TRACK_LENGTH,
    MIN_DIRECTION_STEP_KM,
    ensure_output_dirs,
    load_point_table,
)


# ==========================================================================
# V2风格参数：参考Radar_Tracking_GUI_V2.m调优
# ==========================================================================
# 卡尔曼模型每帧时间步长（与V2一致，按帧计）
TRACK_DT = 1.0
# 速度换算系数（数据vx/vy为km/h，转换为km/帧，1帧=60秒）
TRACK_VEL_SCALE = 60.0  # 除以60得到km/帧
# 过程噪声标准差（赋予目标机动转弯能力）
TRACK_SIG_A = 2.0
# 测量噪声标准差（与V2一致，单位km，25m→0.025km）
TRACK_SIG_Z = 0.025
# 马氏距离波门阈值（卡方2自由度，α≈0.01）
TRACK_GATE_THRESHOLD = 9.21
# 最大允许连续漏检次数（超过则判定目标消失）
TRACK_MAX_MISSED = 4
# 建档位移门限（km，3km以上才算真移动）
TRACK_MIN_DISP_KM = 3.0
# 航迹直线性阈值（过滤往返跳动的海杂波噪声链）
TRACK_MIN_STRAIGHTNESS = 0.5

# 常速模型状态转移矩阵
TRACK_A = np.array(
    [
        [1.0, TRACK_DT, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, TRACK_DT],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
)
# 观测矩阵（只观测位置）
TRACK_H = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=float,
)
# 过程噪声协方差
TRACK_Q = np.array(
    [
        [TRACK_DT**4 / 4, TRACK_DT**3 / 2, 0.0, 0.0],
        [TRACK_DT**3 / 2, TRACK_DT**2, 0.0, 0.0],
        [0.0, 0.0, TRACK_DT**4 / 4, TRACK_DT**3 / 2],
        [0.0, 0.0, TRACK_DT**3 / 2, TRACK_DT**2],
    ],
    dtype=float,
) * (TRACK_SIG_A**2)
# 测量噪声协方差
TRACK_R = np.diag([TRACK_SIG_Z**2, TRACK_SIG_Z**2])
# 状态维度
STATE_DIM = 4

# 候选航迹文件名来源：保存所有关联链路
TRACK_FILE_NAME = "candidate_tracks.csv"
# 确认航迹文件名来源：保存稳定疑似目标航迹
CONFIRMED_TRACK_FILE_NAME = "confirmed_tracks.csv"
# 航迹摘要文件名来源：报告分析引用
TRACK_SUMMARY_FILE_NAME = "track_summary.csv"
# 确认航迹字段来源：空结果也保留稳定表头
TRACK_RESULT_COLUMNS = [
    "frame_idx",
    "track_id",
    "raw_x",
    "raw_y",
    "raw_vx",
    "raw_vy",
    "smooth_x",
    "smooth_y",
    "snr",
    "amp",
    "velocity",
    "range",
    "lon",
    "lat",
    "class_id",
]
# 航迹摘要字段来源：报告分析表格固定读取
TRACK_SUMMARY_COLUMNS = [
    "track_id",
    "frame_count",
    "start_frame",
    "end_frame",
    "mean_snr",
    "mean_velocity",
    "displacement",
    "path_length",
    "straightness",
    "mean_step",
    "max_step",
    "max_turn_angle",
]


def _predict_all(active_tracks: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    """对所有活跃航迹执行一步预测"""
    predictions = []
    for trk in active_tracks:
        predicted_state = TRACK_A @ trk["state"]
        predicted_cov = TRACK_A @ trk["cov"] @ TRACK_A.T + TRACK_Q
        predictions.append((predicted_state, predicted_cov))
    return predictions


def _build_cost_matrix(
    active_tracks: list[dict],
    predictions: list[tuple[np.ndarray, np.ndarray]],
    xy_data: np.ndarray,
) -> np.ndarray:
    """构建全局代价矩阵（马氏距离），不满足波门的置inf"""
    n_tracks = len(active_tracks)
    n_obs = len(xy_data)
    cost = np.full((n_tracks, n_obs), np.inf)
    for j in range(n_tracks):
        pred_state, pred_cov = predictions[j]
        z_pred = TRACK_H @ pred_state
        S_innov = TRACK_H @ pred_cov @ TRACK_H.T + TRACK_R
        try:
            S_inv = np.linalg.inv(S_innov)
        except np.linalg.LinAlgError:
            continue
        for i in range(n_obs):
            innovation = xy_data[i, [0, 2]] - z_pred
            d = float(innovation.T @ S_inv @ innovation)
            if d < TRACK_GATE_THRESHOLD:
                cost[j, i] = d
    return cost


def _greedy_global_assign(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """全局贪婪最小代价关联，返回(assign, used)"""
    n_tracks, n_obs = cost.shape
    assign = np.full(n_tracks, -1, dtype=int)
    used = np.zeros(n_obs, dtype=bool)
    while True:
        idx = np.argmin(cost)
        if np.isinf(cost.flat[idx]):
            break
        j, i = np.unravel_index(idx, cost.shape)
        assign[j] = i
        used[i] = True
        cost[j, :] = np.inf
        cost[:, i] = np.inf
    return assign, used


def _update_matched(
    trk: dict,
    pred_state: np.ndarray,
    pred_cov: np.ndarray,
    z: np.ndarray,
    frame_idx: int,
    raw_row: pd.Series,
    xy_row: np.ndarray,
) -> None:
    """Kalman更新：匹配成功"""
    S_innov = TRACK_H @ pred_cov @ TRACK_H.T + TRACK_R
    K = pred_cov @ TRACK_H.T @ np.linalg.inv(S_innov)
    xu = pred_state + K @ (z - TRACK_H @ pred_state)
    pu = (np.eye(STATE_DIM) - K @ TRACK_H) @ pred_cov
    trk["state"] = xu
    trk["cov"] = pu
    trk["missed"] = 0
    trk["smooth_path"].append([xu[0], xu[2]])
    trk["meas_path"].append([z[0], z[1]])
    trk["frames"].append(frame_idx)
    trk["raw_rows"].append(raw_row.to_dict())


def _update_missed(
    trk: dict,
    pred_state: np.ndarray,
    pred_cov: np.ndarray,
) -> None:
    """Kalman更新：漏检，使用预测值维持"""
    trk["state"] = pred_state
    trk["cov"] = pred_cov
    trk["missed"] += 1
    trk["smooth_path"].append([pred_state[0], pred_state[2]])
    trk["meas_path"].append([np.nan, np.nan])


def _clip_tail_and_validate(trk: dict) -> bool:
    """智能断尾裁剪后判断航迹是否有效
    注意：frames/raw_rows只存真实匹配帧，不含外推帧，
    裁剪smooth_path/meas_path时不对frames/raw_rows操作。"""
    m = trk["missed"]
    if m > 0 and len(trk["meas_path"]) > m:
        trk["smooth_path"] = trk["smooth_path"][:-m]
        trk["meas_path"] = trk["meas_path"][:-m]
    valid_count = sum(~np.isnan(p[0]) for p in trk["meas_path"])
    if valid_count < MIN_TRACK_LENGTH:
        return False
    if len(trk["smooth_path"]) < 2:
        return False
    start = np.array(trk["smooth_path"][0])
    end = np.array(trk["smooth_path"][-1])
    disp = float(np.linalg.norm(end - start))
    if disp <= TRACK_MIN_DISP_KM:
        return False
    # 计算直线性：首尾位移 / 累计路径长度
    coords = np.array(trk["smooth_path"])
    path_len = float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
    straightness = disp / path_len if path_len > 0 else 0.0
    return straightness >= TRACK_MIN_STRAIGHTNESS


def _init_new_track(raw_row: pd.Series, xy_row: np.ndarray, frame_idx: int, track_id: int) -> dict:
    """从未关联点迹初始化新航迹（速度从km/h换算为km/帧）"""
    x, vx_raw, y, vy_raw = xy_row
    vx = vx_raw / TRACK_VEL_SCALE
    vy = vy_raw / TRACK_VEL_SCALE
    return {
        "track_id": track_id,
        "state": np.array([x, vx, y, vy], dtype=float),
        "cov": np.diag([TRACK_SIG_Z**2, 100.0, TRACK_SIG_Z**2, 100.0]),
        "smooth_path": [[x, y]],
        "meas_path": [[x, y]],
        "frames": [frame_idx],
        "raw_rows": [raw_row.to_dict()],
        "missed": 0,
    }


def link_tracks(point_table: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """V2风格全局代价矩阵航迹关联：直接处理原始点迹，含归档系统"""
    active_tracks: list[dict] = []
    archived_tracks: list[dict] = []
    next_id = 1

    for frame_idx in sorted(point_table["frame_idx"].unique()):
        frame_data = point_table[point_table["frame_idx"] == frame_idx].reset_index(drop=True)
        xy_data = frame_data[["x", "vx", "y", "vy"]].to_numpy(dtype=float)
        n_obs = len(xy_data)
        n_tracks = len(active_tracks)

        # 步骤A：预测所有活跃航迹
        predictions = _predict_all(active_tracks) if n_tracks > 0 else []

        # 步骤B：全局代价矩阵 + 贪婪关联
        if n_tracks > 0 and n_obs > 0:
            cost = _build_cost_matrix(active_tracks, predictions, xy_data)
            assign, used = _greedy_global_assign(cost)
        else:
            assign = np.full(n_tracks, -1, dtype=int)
            used = np.zeros(n_obs, dtype=bool)

        # 步骤C：量测更新
        for j in range(n_tracks):
            pred_state, pred_cov = predictions[j]
            if assign[j] >= 0:
                i = int(assign[j])
                z = xy_data[i, [0, 2]]
                _update_matched(active_tracks[j], pred_state, pred_cov, z, frame_idx, frame_data.iloc[i], xy_data[i])
            else:
                _update_missed(active_tracks[j], pred_state, pred_cov)

        # 步骤D：航迹生命周期管理（断尾裁剪 + 建档门限审查）
        keep_indices = []
        for j in range(n_tracks):
            if active_tracks[j]["missed"] < TRACK_MAX_MISSED:
                keep_indices.append(j)
            else:
                if _clip_tail_and_validate(active_tracks[j]):
                    archived_tracks.append(active_tracks[j])
        active_tracks = [active_tracks[i] for i in keep_indices]

        # 步骤E：未关联点迹初始化新航迹
        for i in range(n_obs):
            if not used[i]:
                active_tracks.append(_init_new_track(frame_data.iloc[i], xy_data[i], frame_idx, next_id))
                next_id += 1

    # 循环结束后最终清理：存活到最后的航迹也做断尾审查
    for trk in active_tracks:
        if _clip_tail_and_validate(trk):
            archived_tracks.append(trk)

    # 转换为输出DataFrame（使用smooth_x/smooth_y构建规范列）
    track_table = _build_track_dataframe(archived_tracks)
    return track_table, archived_tracks


def _build_track_dataframe(archived_tracks: list[dict]) -> pd.DataFrame:
    """将归档航迹转换为跟踪结果DataFrame"""
    all_rows = []
    for trk in archived_tracks:
        smooth_arr = np.array(trk["smooth_path"])
        meas_arr = np.array(trk["meas_path"])
        for k, raw_row in enumerate(trk["raw_rows"]):
            all_rows.append(
                {
                    "track_id": trk["track_id"],
                    "frame_idx": trk["frames"][k],
                    "smooth_x": smooth_arr[k, 0],
                    "smooth_y": smooth_arr[k, 1],
                    "meas_x": meas_arr[k, 0] if not np.isnan(meas_arr[k, 0]) else np.nan,
                    "meas_y": meas_arr[k, 1] if not np.isnan(meas_arr[k, 1]) else np.nan,
                    "snr": raw_row.get("snr", np.nan),
                    "amp": raw_row.get("amp", np.nan),
                    "velocity": raw_row.get("velocity", np.nan),
                    "vx": raw_row.get("vx", np.nan),
                    "vy": raw_row.get("vy", np.nan),
                    "range": raw_row.get("range", np.nan),
                    "lon": raw_row.get("lon", np.nan),
                    "lat": raw_row.get("lat", np.nan),
                }
            )
    if not all_rows:
        return pd.DataFrame(columns=["track_id", "frame_idx", "smooth_x", "smooth_y", "meas_x", "meas_y"])
    return pd.DataFrame(all_rows)


def calculate_track_turn_angles(coordinate_values: np.ndarray) -> list[float]:
    """计算整条航迹的有效转向角"""
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


def summarize_tracks(track_table: pd.DataFrame) -> pd.DataFrame:
    """汇总航迹关键指标（基于平滑位置和原始量测）"""
    if track_table.empty:
        return pd.DataFrame(columns=TRACK_SUMMARY_COLUMNS)

    ordered = track_table.sort_values(["track_id", "frame_idx"])
    grouped = ordered.groupby("track_id")

    summary = grouped.agg(
        frame_count=("frame_idx", "count"),
        start_frame=("frame_idx", "min"),
        end_frame=("frame_idx", "max"),
        mean_snr=("snr", "mean"),
        mean_velocity=("velocity", "mean"),
    ).reset_index()

    quality_rows = []
    for track_id, grp in ordered.groupby("track_id"):
        coords = grp[["smooth_x", "smooth_y"]].to_numpy(dtype=float)
        step_distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        path_length = float(step_distances.sum()) if len(step_distances) > 0 else 0.0
        displacement = float(np.linalg.norm(coords[-1] - coords[0])) if len(coords) >= 2 else 0.0
        straightness = displacement / path_length if path_length > 0 else 0.0
        turn_angles = calculate_track_turn_angles(coords)
        quality_rows.append(
            {
                "track_id": track_id,
                "displacement": displacement,
                "path_length": path_length,
                "straightness": straightness,
                "mean_step": float(step_distances.mean()) if len(step_distances) else 0.0,
                "max_step": float(step_distances.max()) if len(step_distances) else 0.0,
                "max_turn_angle": float(max(turn_angles)) if turn_angles else 0.0,
            }
        )

    quality_table = pd.DataFrame(quality_rows)
    summary = summary.merge(quality_table, on="track_id", how="left")
    summary = summary.sort_values(["straightness", "frame_count", "mean_snr"], ascending=False)
    return summary[TRACK_SUMMARY_COLUMNS]


def select_confirmed_tracks(
    track_table: pd.DataFrame, archived_tracks: list[dict]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从归档航迹中选取确认航迹（已通过长度+位移门限）"""
    if track_table.empty:
        empty_cols = ["track_id", "frame_idx", "smooth_x", "smooth_y"]
        return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=TRACK_SUMMARY_COLUMNS)

    track_summary = summarize_tracks(track_table)
    return track_table, track_summary


def rank_tracks(track_summary: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """综合评分排名：straightness × frame_count × mean_snr，选Top N"""
    if track_summary.empty:
        return track_summary
    scored = track_summary.copy()
    scored["quality_score"] = (
        scored["straightness"] * scored["frame_count"] * scored["mean_snr"].clip(lower=1.0)
    )
    scored = scored.sort_values("quality_score", ascending=False)
    return scored.head(top_n)


def main():
    """V2风格目标检测与跟踪主流程"""
    sys.stdout.reconfigure(encoding="utf-8")
    ensure_output_dirs()

    print("=" * 60)
    print("高频地波雷达目标跟踪 — V2全局代价矩阵版本")
    print("=" * 60)

    # 1. 读取原始点迹数据
    point_table = load_point_table()
    # 清洗无效点迹（去掉snr/amp/velocity等关键字段缺失的行）
    clean_table = point_table.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["x", "y", "vx", "vy", "snr", "amp", "velocity"]
    )
    print(f"原始点迹: {len(point_table)} → 清洗后: {len(clean_table)}")
    print(f"帧范围: {int(clean_table['frame_idx'].min())} - {int(clean_table['frame_idx'].max())}")

    # 2. 全局代价矩阵航迹关联（跳过DBSCAN聚类，直接处理原始点迹）
    track_table, archived_tracks = link_tracks(clean_table)
    print(f"归档航迹数量: {len(archived_tracks)}")

    # 3. 生成航迹摘要
    confirmed_tracks, track_summary = select_confirmed_tracks(track_table, archived_tracks)

    # 4. 保存结果
    confirmed_tracks.to_csv(RESULT_DIR / CONFIRMED_TRACK_FILE_NAME, index=False, encoding="utf-8-sig")
    track_summary.to_csv(RESULT_DIR / TRACK_SUMMARY_FILE_NAME, index=False, encoding="utf-8-sig")
    clean_table.to_csv(TABLE_DIR / "strong_points.csv", index=False, encoding="utf-8-sig")

    # 5. 输出运行摘要
    print("-" * 60)
    print(f"确认航迹总数: {track_summary['track_id'].nunique() if not track_summary.empty else 0}")

    # 6. 综合评分排名展示Top航迹
    top_summary = rank_tracks(track_summary, top_n=12)
    if not top_summary.empty:
        print(f"\n{'='*60}")
        print("综合评分 Top 12 优质航迹（straightness × 帧数 × SNR）")
        print(f"{'='*60}")
        display_cols = ["track_id", "frame_count", "displacement", "straightness", "mean_snr"]
        top_display = top_summary[display_cols].copy()
        top_display["track_id"] = top_display["track_id"].astype(int)
        top_display["displacement"] = top_display["displacement"].round(1)
        top_display["straightness"] = top_display["straightness"].round(3)
        top_display["mean_snr"] = top_display["mean_snr"].round(1)
        print(top_display.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
