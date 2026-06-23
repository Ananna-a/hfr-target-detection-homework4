import sys

from hfr_utils import RESULT_DIR, TABLE_DIR, all_frames_to_dataframe, ensure_output_dirs, load_hfr_data


# 统计表文件名来源：报告需要引用整理后的点迹数据
POINT_TABLE_NAME = "整理后点迹数据.csv"
# 帧统计文件名来源：报告需要引用每帧点迹数量
FRAME_STAT_NAME = "每帧点迹统计.csv"
# 信噪比无穷值标记来源：CSV中不适合直接保存无穷值
INVALID_SNR_TEXT = "无效"


def build_frame_stat(point_table):
    # 按帧统计点迹数量和有效信噪比
    valid_snr_table = point_table.replace([float("inf"), float("-inf")], float("nan"))
    frame_stat = (
        valid_snr_table.groupby("frame_idx")
        .agg(
            点迹数量=("id", "count"),
            平均距离=("range", "mean"),
            平均速度=("velocity", "mean"),
            平均信噪比=("snr", "mean"),
            最大信噪比=("snr", "max"),
        )
        .reset_index()
    )
    return frame_stat


def main():
    # 设置控制台编码
    sys.stdout.reconfigure(encoding="utf-8")
    ensure_output_dirs()

    # 读取并整理全部点迹
    hfr_data = load_hfr_data()
    point_table = all_frames_to_dataframe(hfr_data)
    frame_stat = build_frame_stat(point_table)

    # 保存整理后的表格
    point_table_path = TABLE_DIR / POINT_TABLE_NAME
    frame_stat_path = TABLE_DIR / FRAME_STAT_NAME
    point_table.to_csv(point_table_path, index=False, encoding="utf-8-sig")
    frame_stat.to_csv(frame_stat_path, index=False, encoding="utf-8-sig")

    # 输出整理结果
    print(f"整理后点迹数据已保存：{point_table_path}")
    print(f"每帧点迹统计已保存：{frame_stat_path}")
    print(f"总帧数：{hfr_data.shape[1]}")
    print(f"总点迹数：{len(point_table)}")
    print(f"结果目录：{RESULT_DIR}")


if __name__ == "__main__":
    # 执行点迹整理流程
    main()
