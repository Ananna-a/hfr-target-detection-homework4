from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio


# 项目根目录来源：code目录的上一级
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 数据文件来源：课程提供的点迹数据
DATA_FILE = PROJECT_ROOT / "点迹数据" / "HFRData.mat"
# 表格目录来源：保存中间整理数据
TABLE_DIR = PROJECT_ROOT / "report" / "tables"
# 结果目录来源：保存检测与航迹结果
RESULT_DIR = PROJECT_ROOT / "report" / "results"
# 图片目录来源：保存报告主图
FIGURE_DIR = PROJECT_ROOT / "report" / "figures"
# xy字段列索引来源：字段说明中xy为x、vx、y、vy
XY_X_COL = 0
XY_VX_COL = 1
XY_Y_COL = 2
XY_VY_COL = 3
# 速度换算来源：1 m/s = 3.6 km/h
KMH_PER_MPS = 3.6
# 北京时间偏移来源：雷达头字段为本地年月日时分秒
BEIJING_UTC_OFFSET_HOURS = 8
# 帧时间时区来源：统一重建每帧时间戳
FRAME_TIMEZONE = timezone(timedelta(hours=BEIJING_UTC_OFFSET_HOURS))
# 帧时间文本格式来源：报告和表格展示
FRAME_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
# 图片分辨率来源：报告插图清晰度要求
FIGURE_DPI = 300
# Nature风格蓝色来源：候选目标主强调色
BLUE_MAIN = "#0F4D92"
# Nature风格浅蓝来源：候选目标辅助色
BLUE_SECONDARY = "#3775BA"
# Nature风格红色来源：中心或终点强调色
RED_STRONG = "#B64342"
# Nature风格中性浅灰来源：背景点迹
NEUTRAL_LIGHT = "#CFCECE"
# Nature风格中性中灰来源：辅助元素
NEUTRAL_MID = "#767676"
# Nature风格中性深灰来源：文字和坐标轴
NEUTRAL_DARK = "#4D4D4D"
# Nature风格黑色来源：关键线条
NEUTRAL_BLACK = "#272727"
# 航迹颜色来源：避开候选簇蓝色以区分确认航迹
TRACK_COLOR_LIST = ["#B64342", "#2F8F5B", "#9A4D8E", "#0F4D92"]

# 信噪比分位阈值来源：按帧和距离分区保留高信噪比点迹
SNR_QUANTILE = 0.85
# 幅度分位阈值来源：按帧和距离分区保留高幅度点迹
AMP_QUANTILE = 0.65
# 距离分区数量来源：改善不同距离背景强弱差异
LOCAL_RANGE_BIN_COUNT = 6
# 距离分区最少点数来源：避免小样本阈值不稳定
LOCAL_RANGE_MIN_POINTS = 30
# 综合信号评分阈值来源：保留单项很强但未同时过线的点迹
SIGNAL_SCORE_MIN = 1.75
# 聚类空间尺度来源：目标候选邻域经验尺度
SPACE_SCALE_KM = 10.0
# 聚类速度尺度来源：径向速度差经验尺度
VELOCITY_SCALE_KMH = 15.0
# 聚类信噪比尺度来源：有效信噪比分布跨度
SNR_SCALE_DB = 6.0
# 聚类幅度尺度来源：有效幅度分布跨度
AMP_SCALE = 15.0
# DBSCAN邻域半径来源：物理尺度归一化后的经验值
DBSCAN_EPS = 1.10
# DBSCAN最少点数来源：候选点簇稳定性要求
DBSCAN_MIN_SAMPLES = 4
# 候选簇最小点数来源：过滤过小点簇
MIN_CLUSTER_SIZE = 5
# 单帧展示帧来源：该帧有较清晰候选点簇
DISPLAY_FRAME_ID = 42
# 航迹关联距离来源：相邻帧一分钟间隔下的候选中心关联门
MAX_LINK_DISTANCE_KM = 5.0
# 航迹预测时间间隔来源：本数据相邻帧时间间隔为60秒
TRACK_FRAME_INTERVAL_SECONDS = 60.0
# 小时换算来源：速度单位为km/h
SECONDS_PER_HOUR = 3600.0
# 航迹关联速度差来源：相邻帧径向速度最大允许差
MAX_VELOCITY_DIFF_KMH = 15.0
# 航迹关联速度矢量差来源：约束xy速度分量突变
MAX_VECTOR_VELOCITY_DIFF_KMH = 8.0
# 航迹方向最小步长来源：过短中心位移不参与方向判决
MIN_DIRECTION_STEP_KM = 0.25
# 航迹方向变化阈值来源：过滤相邻帧明显反向跳接
MAX_DIRECTION_CHANGE_DEG = 125.0
# 航迹最大断帧来源：本实验只关联相邻帧候选
MAX_TRACK_GAP_FRAMES = 1
# 确认航迹长度来源：保留至少三分钟连续出现的候选
MIN_TRACK_LENGTH = 4
# 确认航迹直线性来源：过滤往返跳动明显的候选链
MIN_TRACK_STRAIGHTNESS = 0.60
# 确认航迹最大步长来源：过滤相邻帧中心突跳
MAX_TRACK_STEP_KM = 2.00
# 确认航迹平均步长来源：过滤整体跳动偏大的候选链
MAX_MEAN_TRACK_STEP_KM = 1.00


def ensure_output_dirs() -> None:
    # 创建输出目录
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def configure_plot_style() -> None:
    # 配置统一中文绘图风格
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#222222"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.titlesize"] = 9
    plt.rcParams["axes.labelsize"] = 7.5
    plt.rcParams["xtick.labelsize"] = 6.8
    plt.rcParams["ytick.labelsize"] = 6.8
    plt.rcParams["legend.fontsize"] = 6.8
    plt.rcParams["grid.color"] = "#dddddd"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.55
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def save_figure(figure: plt.Figure, file_name: str) -> Path:
    # 保存报告图片和矢量文件
    ensure_output_dirs()
    figure_path = FIGURE_DIR / file_name
    figure.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    if figure_path.suffix.lower() == ".png":
        figure.savefig(figure_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
        figure.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return figure_path


def load_hfr_data() -> np.ndarray:
    # 读取MAT文件中的HFRData变量
    mat_data = sio.loadmat(DATA_FILE)
    return mat_data["HFRData"]


def get_frame_struct(hfr_data: np.ndarray, frame_idx: int) -> np.void:
    # 获取指定帧结构体
    return hfr_data[0, frame_idx][0, 0]


def get_header_value(frame_struct: np.void, field_name: str) -> float:
    # 读取帧头字段
    header_struct = frame_struct["header"][0, 0]
    return float(np.asarray(header_struct[field_name]).ravel()[0])


def get_header_datetime(frame_struct: np.void) -> datetime:
    # 重建帧头年月日时分秒时间
    year = int(get_header_value(frame_struct, "dyear"))
    month = int(get_header_value(frame_struct, "dmonth"))
    day = int(get_header_value(frame_struct, "dday"))
    hour = int(get_header_value(frame_struct, "dhour"))
    minute = int(get_header_value(frame_struct, "dminute"))
    second = int(get_header_value(frame_struct, "dsec"))
    return datetime(year, month, day, hour, minute, second, tzinfo=FRAME_TIMEZONE)


def frame_to_dataframe(hfr_data: np.ndarray, frame_idx: int) -> pd.DataFrame:
    # 转换单帧点迹为表格
    frame_struct = get_frame_struct(hfr_data, frame_idx)
    xy_values = np.asarray(frame_struct["xy"], dtype=float)
    point_count = int(np.asarray(frame_struct["PlotCnt"]).ravel()[0])
    raw_unixtime = get_header_value(frame_struct, "unixtime")
    frame_datetime = get_header_datetime(frame_struct)
    frame_timestamp = frame_datetime.timestamp()
    frame_time_text = frame_datetime.strftime(FRAME_TIME_FORMAT)
    velocity_values = np.asarray(frame_struct["velocity"]).reshape(-1).astype(float)

    return pd.DataFrame(
        {
            "frame_idx": np.full(point_count, frame_idx + 1, dtype=int),
            "time": np.full(point_count, frame_timestamp, dtype=float),
            "frame_time": np.full(point_count, frame_time_text),
            "raw_unixtime": np.full(point_count, raw_unixtime, dtype=float),
            "id": np.asarray(frame_struct["id"]).reshape(-1).astype(int),
            "range": np.asarray(frame_struct["range"]).reshape(-1).astype(float),
            "velocity": velocity_values,
            "velocity_mps": velocity_values / KMH_PER_MPS,
            "ang": np.asarray(frame_struct["ang"]).reshape(-1).astype(float),
            "x": xy_values[:, XY_X_COL],
            "vx": xy_values[:, XY_VX_COL],
            "y": xy_values[:, XY_Y_COL],
            "vy": xy_values[:, XY_VY_COL],
            "lon": np.asarray(frame_struct["lon"]).reshape(-1).astype(float),
            "lat": np.asarray(frame_struct["lat"]).reshape(-1).astype(float),
            "amp": np.asarray(frame_struct["Amp"]).reshape(-1).astype(float),
            "rcs": np.asarray(frame_struct["RCS"]).reshape(-1).astype(float),
            "snr": np.asarray(frame_struct["snr"]).reshape(-1).astype(float),
            "class_id": np.asarray(frame_struct["Class"]).reshape(-1).astype(int),
            "flag": np.asarray(frame_struct["flag"]).reshape(-1).astype(int),
        }
    )


def load_point_table() -> pd.DataFrame:
    # 读取或生成整理后的点迹表
    point_table_path = TABLE_DIR / "points_clean.csv"
    if point_table_path.exists():
        return pd.read_csv(point_table_path)
    hfr_data = load_hfr_data()
    point_tables = [frame_to_dataframe(hfr_data, frame_idx) for frame_idx in range(hfr_data.shape[1])]
    return pd.concat(point_tables, ignore_index=True)
