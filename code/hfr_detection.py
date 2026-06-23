import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from hfr_utils import (
    RESULT_DIR,
    all_frames_to_dataframe,
    configure_chinese_plot,
    ensure_output_dirs,
    load_hfr_data,
    save_figure,
)


# 信噪比分位数来源：每帧保留相对较强回波点
SNR_QUANTILE = 0.80
# 幅度分位数来源：每帧保留相对较强幅度点
AMP_QUANTILE = 0.55
# DBSCAN邻域半径来源：标准化特征空间中的经验参数
DBSCAN_EPS = 0.85
# DBSCAN最小点数来源：避免孤立点形成目标
DBSCAN_MIN_SAMPLES = 5
# 候选簇最小点数来源：报告中目标候选需要稳定点簇
MIN_CLUSTER_SIZE = 5
# 检测图展示帧来源：优先展示第1帧的检测效果
DISPLAY_FRAME_ID = 1
# 散点大小来源：检测图需要区分原始点和候选点
POINT_SIZE = 10
# 簇中心点大小来源：突出目标候选中心
CENTER_SIZE = 75
# 图像宽度来源：报告图片尺寸
FIGURE_WIDTH = 10
# 图像高度来源：报告图片尺寸
FIGURE_HEIGHT = 8
# 候选点输出文件名来源：保存聚类后的点级结果
CANDIDATE_POINT_FILE_NAME = "目标候选点.csv"
# 候选簇输出文件名来源：保存每个候选目标摘要
CANDIDATE_CLUSTER_FILE_NAME = "目标候选簇.csv"


def get_detection_input(frame_table: pd.DataFrame) -> pd.DataFrame:
    # 根据有效信噪比和自适应门限筛选候选输入点
    valid_table = frame_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["snr", "amp", "x", "y", "velocity"])
    if valid_table.empty:
        return valid_table

    snr_threshold = valid_table["snr"].quantile(SNR_QUANTILE)
    amp_threshold = valid_table["amp"].quantile(AMP_QUANTILE)
    detection_input = valid_table[
        (valid_table["snr"] >= snr_threshold)
        & (valid_table["amp"] >= amp_threshold)
    ].copy()
    return detection_input


def cluster_frame(frame_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 对单帧候选输入点进行DBSCAN聚类
    detection_input = get_detection_input(frame_table)
    if len(detection_input) < DBSCAN_MIN_SAMPLES:
        return detection_input.assign(cluster_id=-1), pd.DataFrame()

    feature_table = detection_input[["x", "y", "velocity", "snr", "amp"]]
    scaled_feature = StandardScaler().fit_transform(feature_table)
    cluster_model = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    detection_input["cluster_id"] = cluster_model.fit_predict(scaled_feature)

    # 汇总每个有效簇的中心和统计特征
    cluster_rows = []
    valid_cluster_ids = sorted(cluster_id for cluster_id in detection_input["cluster_id"].unique() if cluster_id >= 0)
    for cluster_id in valid_cluster_ids:
        cluster_table = detection_input[detection_input["cluster_id"] == cluster_id]
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
                "time": float(cluster_table["time"].mean()),
                "mean_velocity": float(cluster_table["velocity"].mean()),
                "mean_snr": float(cluster_table["snr"].mean()),
                "max_snr": float(cluster_table["snr"].max()),
                "mean_amp": float(cluster_table["amp"].mean()),
                "score": float(cluster_table["snr"].mean() + np.log1p(len(cluster_table))),
            }
        )

    cluster_summary = pd.DataFrame(cluster_rows)
    return detection_input, cluster_summary


def detect_all_frames(point_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 对全部帧执行目标候选检测
    candidate_point_tables = []
    candidate_cluster_tables = []
    for frame_id, frame_table in point_table.groupby("frame_idx"):
        candidate_points, cluster_summary = cluster_frame(frame_table)
        candidate_point_tables.append(candidate_points)
        if not cluster_summary.empty:
            candidate_cluster_tables.append(cluster_summary)

    all_candidate_points = pd.concat(candidate_point_tables, ignore_index=True)
    if candidate_cluster_tables:
        all_candidate_clusters = pd.concat(candidate_cluster_tables, ignore_index=True)
    else:
        all_candidate_clusters = pd.DataFrame()
    return all_candidate_points, all_candidate_clusters


def plot_detection_result(point_table: pd.DataFrame, candidate_points: pd.DataFrame, candidate_clusters: pd.DataFrame):
    # 绘制指定帧目标候选检测结果
    frame_table = point_table[point_table["frame_idx"] == DISPLAY_FRAME_ID]
    frame_candidate_points = candidate_points[candidate_points["frame_idx"] == DISPLAY_FRAME_ID]
    frame_candidate_clusters = candidate_clusters[candidate_clusters["frame_idx"] == DISPLAY_FRAME_ID]

    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    axis.scatter(frame_table["x"], frame_table["y"], s=POINT_SIZE, color="#bdbdbd", alpha=0.45, label="原始点迹")

    valid_cluster_points = frame_candidate_points[frame_candidate_points["cluster_id"] >= 0]
    if not valid_cluster_points.empty:
        scatter = axis.scatter(
            valid_cluster_points["x"],
            valid_cluster_points["y"],
            c=valid_cluster_points["cluster_id"],
            s=POINT_SIZE * 2,
            cmap="tab20",
            alpha=0.9,
            label="候选簇点迹",
        )
        figure.colorbar(scatter, ax=axis, label="候选簇编号")

    if not frame_candidate_clusters.empty:
        axis.scatter(
            frame_candidate_clusters["center_x"],
            frame_candidate_clusters["center_y"],
            s=CENTER_SIZE,
            color="red",
            marker="x",
            linewidths=2.2,
            label="候选目标中心",
        )

    axis.set_title(f"第{DISPLAY_FRAME_ID}帧目标候选检测结果")
    axis.set_xlabel("x坐标/km")
    axis.set_ylabel("y坐标/km")
    axis.grid(True, linestyle="--", alpha=0.35)
    axis.legend(loc="best")
    axis.set_aspect("equal", adjustable="box")
    return save_figure(figure, f"第{DISPLAY_FRAME_ID}帧目标候选检测结果.png")


def main():
    # 设置控制台编码和中文绘图
    sys.stdout.reconfigure(encoding="utf-8")
    configure_chinese_plot()
    ensure_output_dirs()

    # 读取点迹并执行检测
    hfr_data = load_hfr_data()
    point_table = all_frames_to_dataframe(hfr_data)
    candidate_points, candidate_clusters = detect_all_frames(point_table)

    # 保存检测结果
    candidate_point_path = RESULT_DIR / CANDIDATE_POINT_FILE_NAME
    candidate_cluster_path = RESULT_DIR / CANDIDATE_CLUSTER_FILE_NAME
    candidate_points.to_csv(candidate_point_path, index=False, encoding="utf-8-sig")
    candidate_clusters.to_csv(candidate_cluster_path, index=False, encoding="utf-8-sig")
    figure_path = plot_detection_result(point_table, candidate_points, candidate_clusters)

    # 输出检测摘要
    print(f"目标候选点已保存：{candidate_point_path}")
    print(f"目标候选簇已保存：{candidate_cluster_path}")
    print(f"检测结果图已保存：{figure_path}")
    print(f"候选点数量：{len(candidate_points)}")
    print(f"候选簇数量：{len(candidate_clusters)}")


if __name__ == "__main__":
    # 执行目标候选检测流程
    main()
