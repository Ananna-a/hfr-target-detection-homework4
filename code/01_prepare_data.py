import sys

import numpy as np
import pandas as pd

from hfr_config import TABLE_DIR, ensure_output_dirs, load_hfr_data, frame_to_dataframe


# 整理点迹文件名来源：后续检测和绘图统一读取
POINT_TABLE_FILE_NAME = "points_clean.csv"
# 数据摘要文件名来源：报告结果分析引用
DATA_SUMMARY_FILE_NAME = "data_summary.csv"


def build_point_table():
    # 合并所有帧点迹数据
    hfr_data = load_hfr_data()
    frame_tables = [frame_to_dataframe(hfr_data, frame_idx) for frame_idx in range(hfr_data.shape[1])]
    point_table = pd.concat(frame_tables, ignore_index=True)
    return point_table


def build_data_summary(point_table):
    # 统计数据规模和主要字段范围
    valid_snr = point_table["snr"].replace([np.inf, -np.inf], np.nan).dropna()
    summary_rows = [
        {"指标": "帧数", "数值": int(point_table["frame_idx"].nunique())},
        {"指标": "总点迹数", "数值": int(len(point_table))},
        {"指标": "每帧最少点迹数", "数值": int(point_table.groupby("frame_idx").size().min())},
        {"指标": "每帧最多点迹数", "数值": int(point_table.groupby("frame_idx").size().max())},
        {"指标": "距离范围/km", "数值": f"{point_table['range'].min():.2f} - {point_table['range'].max():.2f}"},
        {"指标": "速度范围/(km/h)", "数值": f"{point_table['velocity'].min():.2f} - {point_table['velocity'].max():.2f}"},
        {"指标": "有效信噪比范围/dB", "数值": f"{valid_snr.min():.2f} - {valid_snr.max():.2f}"},
        {"指标": "Class唯一值", "数值": ",".join(str(value) for value in sorted(point_table["class_id"].unique()))},
    ]
    return pd.DataFrame(summary_rows)


def main():
    # 设置控制台编码和输出目录
    sys.stdout.reconfigure(encoding="utf-8")
    ensure_output_dirs()

    # 生成点迹表和摘要表
    point_table = build_point_table()
    data_summary = build_data_summary(point_table)

    # 保存中间数据
    point_table_path = TABLE_DIR / POINT_TABLE_FILE_NAME
    data_summary_path = TABLE_DIR / DATA_SUMMARY_FILE_NAME
    point_table.to_csv(point_table_path, index=False, encoding="utf-8-sig")
    data_summary.to_csv(data_summary_path, index=False, encoding="utf-8-sig")

    # 输出运行摘要
    print(f"点迹整理表已保存：{point_table_path}")
    print(f"数据摘要表已保存：{data_summary_path}")
    print(f"总帧数：{point_table['frame_idx'].nunique()}")
    print(f"总点迹数：{len(point_table)}")


if __name__ == "__main__":
    # 执行数据准备流程
    main()
