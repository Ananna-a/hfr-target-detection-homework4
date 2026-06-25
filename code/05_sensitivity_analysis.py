import importlib.util
import sys
from pathlib import Path

import pandas as pd


# 项目根目录来源：code目录的上一级
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 代码目录来源：加载数字开头的流程脚本
CODE_DIR = PROJECT_ROOT / "code"
# 检测脚本路径来源：复用主流程函数
TRACK_MODULE_PATH = CODE_DIR / "02_detect_and_track.py"
# 敏感性结果文件来源：报告参数分析引用
SENSITIVITY_FILE = PROJECT_ROOT / "report" / "results" / "sensitivity_analysis.csv"
# 默认参数来源：当前最终方案
DEFAULT_PARAMETERS = {
    "SNR_QUANTILE": 0.80,
    "AMP_QUANTILE": 0.58,
    "SIGNAL_SCORE_MIN": 1.75,
    "DBSCAN_EPS": 1.20,
    "DBSCAN_MIN_SAMPLES": 3,
    "MIN_CLUSTER_SIZE": 4,
    "MAX_LINK_DISTANCE_KM": 4.0,
    "MAX_VELOCITY_DIFF_KMH": 12.0,
    "MAX_VECTOR_VELOCITY_DIFF_KMH": 6.0,
    "MAX_DIRECTION_CHANGE_DEG": 125.0,
    "MAX_TRACK_GAP_FRAMES": 3,
    "MIN_TRACK_LENGTH": 4,
    "MIN_TRACK_STRAIGHTNESS": 0.60,
    "MAX_TRACK_STEP_KM": 2.00,
    "MAX_MEAN_TRACK_STEP_KM": 1.00,
    "MAX_STEP_VELOCITY_KMH": 55.0,
}
# 参数方案来源：对应单帧筛选、聚类、关联、长度和质量确认阶段
SENSITIVITY_SCENARIOS = [
    {
        "scenario_name": "最终方案",
        "stage_name": "基准",
        "changed_parameters": "当前报告参数",
        "overrides": {},
    },
    {
        "scenario_name": "强点筛选偏严",
        "stage_name": "单帧筛选",
        "changed_parameters": "SNR=0.85, Amp=0.65, Score=1.85",
        "overrides": {"SNR_QUANTILE": 0.85, "AMP_QUANTILE": 0.65, "SIGNAL_SCORE_MIN": 1.85},
    },
    {
        "scenario_name": "强点筛选偏松",
        "stage_name": "单帧筛选",
        "changed_parameters": "SNR=0.75, Amp=0.52, Score=1.55",
        "overrides": {"SNR_QUANTILE": 0.75, "AMP_QUANTILE": 0.52, "SIGNAL_SCORE_MIN": 1.55},
    },
    {
        "scenario_name": "聚类邻域偏小",
        "stage_name": "单帧聚类",
        "changed_parameters": "eps=1.10, min_samples=4, min_cluster=5",
        "overrides": {"DBSCAN_EPS": 1.10, "DBSCAN_MIN_SAMPLES": 4, "MIN_CLUSTER_SIZE": 5},
    },
    {
        "scenario_name": "聚类邻域偏大",
        "stage_name": "单帧聚类",
        "changed_parameters": "eps=1.35, min_samples=2, min_cluster=3",
        "overrides": {"DBSCAN_EPS": 1.35, "DBSCAN_MIN_SAMPLES": 2, "MIN_CLUSTER_SIZE": 3},
    },
    {
        "scenario_name": "关联门限偏严",
        "stage_name": "多帧关联",
        "changed_parameters": "距离=3.0, 速度差=10, 矢量差=4.5, 转向=105",
        "overrides": {
            "MAX_LINK_DISTANCE_KM": 3.0,
            "MAX_VELOCITY_DIFF_KMH": 10.0,
            "MAX_VECTOR_VELOCITY_DIFF_KMH": 4.5,
            "MAX_DIRECTION_CHANGE_DEG": 105.0,
        },
    },
    {
        "scenario_name": "关联门限偏松",
        "stage_name": "多帧关联",
        "changed_parameters": "距离=5.5, 速度差=15, 矢量差=8.0, 转向=145",
        "overrides": {
            "MAX_LINK_DISTANCE_KM": 5.5,
            "MAX_VELOCITY_DIFF_KMH": 15.0,
            "MAX_VECTOR_VELOCITY_DIFF_KMH": 8.0,
            "MAX_DIRECTION_CHANGE_DEG": 145.0,
        },
    },
    {
        "scenario_name": "断帧容忍偏小",
        "stage_name": "缓存策略",
        "changed_parameters": "max_gap=1",
        "overrides": {"MAX_TRACK_GAP_FRAMES": 1},
    },
    {
        "scenario_name": "断帧容忍偏大",
        "stage_name": "缓存策略",
        "changed_parameters": "max_gap=4",
        "overrides": {"MAX_TRACK_GAP_FRAMES": 4},
    },
    {
        "scenario_name": "长度要求偏短",
        "stage_name": "长度判决",
        "changed_parameters": "min_length=3",
        "overrides": {"MIN_TRACK_LENGTH": 3},
    },
    {
        "scenario_name": "长度要求偏长",
        "stage_name": "长度判决",
        "changed_parameters": "min_length=5",
        "overrides": {"MIN_TRACK_LENGTH": 5},
    },
    {
        "scenario_name": "质量确认偏严",
        "stage_name": "质量确认",
        "changed_parameters": "直线性=0.70, 最大步长=1.50, 平均步长=0.80, 步进速度=40, 转向=105",
        "overrides": {
            "MIN_TRACK_STRAIGHTNESS": 0.70,
            "MAX_TRACK_STEP_KM": 1.50,
            "MAX_MEAN_TRACK_STEP_KM": 0.80,
            "MAX_STEP_VELOCITY_KMH": 40.0,
            "MAX_DIRECTION_CHANGE_DEG": 105.0,
        },
    },
    {
        "scenario_name": "质量确认偏松",
        "stage_name": "质量确认",
        "changed_parameters": "直线性=0.50, 最大步长=2.50, 平均步长=1.30, 步进速度=80, 转向=145",
        "overrides": {
            "MIN_TRACK_STRAIGHTNESS": 0.50,
            "MAX_TRACK_STEP_KM": 2.50,
            "MAX_MEAN_TRACK_STEP_KM": 1.30,
            "MAX_STEP_VELOCITY_KMH": 80.0,
            "MAX_DIRECTION_CHANGE_DEG": 145.0,
        },
    },
]


def main() -> None:
    # 执行分层参数敏感性分析
    sys.stdout.reconfigure(encoding="utf-8")
    track_module = load_track_module()
    point_table = track_module.load_point_table()
    sensitivity_table = run_sensitivity_analysis(track_module, point_table)
    SENSITIVITY_FILE.parent.mkdir(parents=True, exist_ok=True)
    sensitivity_table.to_csv(SENSITIVITY_FILE, index=False, encoding="utf-8-sig")
    print(f"敏感性分析结果已保存：{SENSITIVITY_FILE}")
    print(sensitivity_table.to_string(index=False))


def load_track_module():
    # 加载检测跟踪脚本
    sys.path.insert(0, str(CODE_DIR))
    module_spec = importlib.util.spec_from_file_location("detect_and_track_module", TRACK_MODULE_PATH)
    track_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(track_module)
    return track_module


def apply_parameters(track_module, parameter_values: dict) -> None:
    # 应用指定参数组合
    for parameter_name, parameter_value in parameter_values.items():
        setattr(track_module, parameter_name, parameter_value)


def restore_default_parameters(track_module) -> None:
    # 恢复默认参数组合
    apply_parameters(track_module, DEFAULT_PARAMETERS)


def get_confirmed_track_labels(confirmed_tracks: pd.DataFrame) -> str:
    # 生成确认航迹编号文本
    if confirmed_tracks.empty:
        return "-"
    track_ids = sorted(confirmed_tracks["track_id"].astype(int).unique())
    return ",".join(f"T{track_id}" for track_id in track_ids)


def run_single_scenario(track_module, point_table: pd.DataFrame, scenario: dict) -> dict:
    # 运行单组参数方案
    restore_default_parameters(track_module)
    scenario_parameters = DEFAULT_PARAMETERS | scenario["overrides"]
    apply_parameters(track_module, scenario_parameters)

    strong_point_table, cluster_table = track_module.detect_all_clusters(point_table)
    track_table = track_module.link_tracks(cluster_table)
    confirmed_tracks, _, length_summary = track_module.select_confirmed_tracks(track_table)

    return {
        "方案": scenario["scenario_name"],
        "阶段": scenario["stage_name"],
        "改变参数": scenario["changed_parameters"],
        "强点数": int(len(strong_point_table)),
        "候选簇": int(len(cluster_table)),
        "候选航迹": int(track_table["track_id"].nunique()) if not track_table.empty else 0,
        "长度达标": int(length_summary["track_id"].nunique()) if not length_summary.empty else 0,
        "质量确认": int(confirmed_tracks["track_id"].nunique()) if not confirmed_tracks.empty else 0,
        "确认航迹": get_confirmed_track_labels(confirmed_tracks),
    }


def run_sensitivity_analysis(track_module, point_table: pd.DataFrame) -> pd.DataFrame:
    # 汇总全部参数方案结果
    scenario_rows = []
    for scenario in SENSITIVITY_SCENARIOS:
        scenario_rows.append(run_single_scenario(track_module, point_table, scenario))
    restore_default_parameters(track_module)
    return pd.DataFrame(scenario_rows)


if __name__ == "__main__":
    # 启动敏感性分析流程
    main()
