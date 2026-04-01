import os
import platform
import sys
import time
import torch


def write_runtime_statistics(start_time, before_shap_time, end_time,
                             SHAP_ANALYSIS_ENABLED, output_source_PandO_folder, gpu_id):
    """
    计算并写入程序运行时间统计信息

    参数:
        start_time: 程序开始时间的时间戳
        before_shap_time: 第10部分开始前的时间戳
        end_time: 程序结束时间的时间戳
        SHAP_ANALYSIS_ENABLED: 是否启用了SHAP分析
        output_source_PandO_folder: 输出文件夹路径
        gpu_id: 使用的GPU编号
    """
    # 计算各部分运行时间
    total_time = end_time - start_time
    before_shap_duration = before_shap_time - start_time
    shap_duration = end_time - before_shap_time if SHAP_ANALYSIS_ENABLED else 0

    # 创建时间统计文件路径
    runtime_stats_path = os.path.join(output_source_PandO_folder, 'runtime_statistics.txt')

    # 写入时间统计信息
    with open(runtime_stats_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("程序运行时间统计\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"第10部分之前的总运行时间: {before_shap_duration:.2f}秒 ({before_shap_duration / 60:.2f}分钟)\n")

        if SHAP_ANALYSIS_ENABLED:
            f.write(f"第10部分(SHAP分析)运行时间: {shap_duration:.2f}秒 ({shap_duration / 60:.2f}分钟)\n")

        f.write(f"程序总运行时间: {total_time:.2f}秒 ({total_time / 60:.2f}分钟)\n\n")

        # 添加详细的时间点信息
        f.write("详细时间点:\n")
        f.write(f"- 程序开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        f.write(f"- 第10部分开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(before_shap_time))}\n")
        f.write(f"- 程序结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")

        f.write("\n" + "=" * 50)

    # 在控制台打印时间统计信息
    print("\n" + "=" * 50)
    print("运行时间统计:")
    print(f"第10部分之前的总运行时间: {before_shap_duration:.2f}秒 ({before_shap_duration / 60:.2f}分钟)")
    if SHAP_ANALYSIS_ENABLED:
        print(f"第10部分(SHAP分析)运行时间: {shap_duration:.2f}秒 ({shap_duration / 60:.2f}分钟)")
    print(f"程序总运行时间: {total_time:.2f}秒 ({total_time / 60:.2f}分钟)")
    print(f"时间统计已保存至: {runtime_stats_path}")
    print("=" * 50)