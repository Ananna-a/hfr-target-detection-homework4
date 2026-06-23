from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio


# 项目根目录来源：当前脚本所在 code 目录的上一级
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 数据文件来源：课程提供的高频地波雷达点迹数据
DATA_FILE = PROJECT_ROOT / "点迹数据" / "HFRData.mat"
# 图片输出目录来源：报告需要插图
FIGURE_DIR = PROJECT_ROOT / "report" / "figures"
# 表格输出目录来源：报告需要统计表
TABLE_DIR = PROJECT_ROOT / "report" / "tables"
# 结果输出目录来源：检测和轨迹结果
RESULT_DIR = PROJECT_ROOT / "report" / "results"
# xy字段列索引来源：字段说明中的x、vx、y、vy顺序
XY_X_COL = 0
XY_VX_COL = 1
XY_Y_COL = 2
XY_VY_COL = 3
# 米每秒转换来源：1 km/h = 1/3.6 m/s
KMH_TO_MPS = 3.6


def configure_chinese_plot() -> None:
    # 配置中文字体和负号显示
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def ensure_output_dirs() -> None:
    # 创建报告输出目录
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def load_hfr_data() -> np.ndarray:
    # 读取MAT文件中的HFRData变量
    mat_data = sio.loadmat(DATA_FILE)
    return mat_data["HFRData"]


def get_frame_struct(hfr_data: np.ndarray, frame_idx: int) -> np.void:
    # 获取指定帧的结构体数据
    return hfr_data[0, frame_idx][0, 0]


def get_header_value(frame_struct: np.void, field_name: str) -> float:
    # 读取header中的单个字段值
    header_struct = frame_struct["header"][0, 0]
    return float(np.asarray(header_struct[field_name]).ravel()[0])


def frame_to_dataframe(hfr_data: np.ndarray, frame_idx: int) -> pd.DataFrame:
    # 将单帧点迹结构体转换为表格
    frame_struct = get_frame_struct(hfr_data, frame_idx)
    xy_values = np.asarray(frame_struct["xy"], dtype=float)
    point_count = int(np.asarray(frame_struct["PlotCnt"]).ravel()[0])
    unix_time = get_header_value(frame_struct, "unixtime")

    data_frame = pd.DataFrame(
        {
            "frame_idx": np.full(point_count, frame_idx + 1, dtype=int),
            "time": np.full(point_count, unix_time, dtype=float),
            "id": np.asarray(frame_struct["id"]).reshape(-1).astype(int),
            "prop": np.asarray(frame_struct["prop"]).reshape(-1).astype(int),
            "range": np.asarray(frame_struct["range"]).reshape(-1).astype(float),
            "velocity": np.asarray(frame_struct["velocity"]).reshape(-1).astype(float),
            "velocity_mps": np.asarray(frame_struct["velocity"]).reshape(-1).astype(float) / KMH_TO_MPS,
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
    return data_frame


def all_frames_to_dataframe(hfr_data: np.ndarray) -> pd.DataFrame:
    # 合并所有帧的点迹表格
    frame_tables = [frame_to_dataframe(hfr_data, frame_idx) for frame_idx in range(hfr_data.shape[1])]
    return pd.concat(frame_tables, ignore_index=True)


def save_figure(figure: plt.Figure, file_name: str) -> Path:
    # 保存报告图片
    ensure_output_dirs()
    figure_path = FIGURE_DIR / file_name
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return figure_path
