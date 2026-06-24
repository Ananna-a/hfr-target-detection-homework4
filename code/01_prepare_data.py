import sys

import numpy as np
import pandas as pd

from hfr_config import TABLE_DIR, ensure_output_dirs, load_hfr_data, frame_to_dataframe


# 整理点迹文件名来源：后续检测和绘图统一读取
POINT_TABLE_FILE_NAME = "points_clean.csv"
# 数据摘要文件名来源：报告结果分析引用
DATA_SUMMARY_FILE_NAME = "data_summary.csv"
# 每帧统计文件名来源：报告数据质量审查引用
FRAME_SUMMARY_FILE_NAME = "frame_summary.csv"
# 清洗必要字段来源：检测和绘图流程的基础输入
REQUIRED_CLEAN_COLUMNS = ["snr", "amp", "x", "y", "velocity", "range", "lon", "lat"]


def build_raw_point_table() -> pd.DataFrame:
    # 合并所有帧点迹数据
    hfr_data = load_hfr_data()
    frame_tables = [frame_to_dataframe(hfr_data, frame_idx) for frame_idx in range(hfr_data.shape[1])]
    point_table = pd.concat(frame_tables, ignore_index=True)
    return point_table


def clean_point_table(raw_point_table: pd.DataFrame) -> pd.DataFrame:
    # 清洗无效数值点迹
    finite_point_table = raw_point_table.replace([np.inf, -np.inf], np.nan)
    point_table = finite_point_table.dropna(subset=REQUIRED_CLEAN_COLUMNS).copy()
    return point_table


def build_data_summary(raw_point_table: pd.DataFrame, point_table: pd.DataFrame) -> pd.DataFrame:
    # 统计数据规模和主要字段范围
    invalid_snr_count = int((~np.isfinite(raw_point_table["snr"])).sum())
    valid_snr = point_table["snr"].dropna()
    summary_rows = [
        {"指标": "帧数", "数值": int(raw_point_table["frame_idx"].nunique())},
        {"指标": "原始总点迹数", "数值": int(len(raw_point_table))},
        {"指标": "清洗后点迹数", "数值": int(len(point_table))},
        {"指标": "无效信噪比点数", "数值": invalid_snr_count},
        {"指标": "每帧最少点迹数", "数值": int(point_table.groupby("frame_idx").size().min())},
        {"指标": "每帧最多点迹数", "数值": int(point_table.groupby("frame_idx").size().max())},
        {"指标": "距离范围/km", "数值": f"{point_table['range'].min():.2f} - {point_table['range'].max():.2f}"},
        {"指标": "速度范围/(km/h)", "数值": f"{point_table['velocity'].min():.2f} - {point_table['velocity'].max():.2f}"},
        {"指标": "有效信噪比范围/dB", "数值": f"{valid_snr.min():.2f} - {valid_snr.max():.2f}"},
        {"指标": "Class唯一值", "数值": ",".join(str(value) for value in sorted(raw_point_table["class_id"].unique()))},
    ]
    return pd.DataFrame(summary_rows)


def build_frame_summary(point_table: pd.DataFrame) -> pd.DataFrame:
    # 统计每帧点迹质量指标
    frame_summary = (
        point_table.groupby("frame_idx")
        .agg(
            time=("time", "first"),
            point_count=("id", "count"),
            range_min=("range", "min"),
            range_max=("range", "max"),
            mean_velocity=("velocity", "mean"),
            mean_snr=("snr", "mean"),
            mean_amp=("amp", "mean"),
        )
        .reset_index()
    )
    return frame_summary


def main():
    # 设置控制台编码和输出目录
    sys.stdout.reconfigure(encoding="utf-8")
    ensure_output_dirs()

    # 生成清洗点迹表和统计表
    raw_point_table = build_raw_point_table()
    point_table = clean_point_table(raw_point_table)
    data_summary = build_data_summary(raw_point_table, point_table)
    frame_summary = build_frame_summary(point_table)

    # 保存中间数据
    point_table_path = TABLE_DIR / POINT_TABLE_FILE_NAME
    data_summary_path = TABLE_DIR / DATA_SUMMARY_FILE_NAME
    frame_summary_path = TABLE_DIR / FRAME_SUMMARY_FILE_NAME
    point_table.to_csv(point_table_path, index=False, encoding="utf-8-sig")
    data_summary.to_csv(data_summary_path, index=False, encoding="utf-8-sig")
    frame_summary.to_csv(frame_summary_path, index=False, encoding="utf-8-sig")

    # 输出运行摘要
    print(f"点迹整理表已保存：{point_table_path}")
    print(f"数据摘要表已保存：{data_summary_path}")
    print(f"每帧统计表已保存：{frame_summary_path}")
    print(f"总帧数：{point_table['frame_idx'].nunique()}")
    print(f"清洗后点迹数：{len(point_table)}")


if __name__ == "__main__":
    # 执行数据准备流程
    main()
