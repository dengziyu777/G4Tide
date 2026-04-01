import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'     # 告诉 OpenMP 运行时允许加载多个副本
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from source.Fv6_generate_time_ticks import Fv6_generate_time_ticks


def Fv6_validate_and_plot(batch1_data, batch2_data, output_folder, title_prefix,
                          batch3_data_folder_path, ylabel, plot_common_period_only, forward_hours,
                          time_interval_hours, rotation_user, print_on = True):
    """
    验证预报和观测数据，并为每个站点单独绘制对比图并保存
    250715：标题中，以batch3_data_folder_path下各文件来命名

    参数:
        batch1_data: 预报数据 (列表，每个元素是一个站点的数据数组 [时间戳, 值])
        batch2_data: 观测数据 (列表，每个元素是一个站点的数据数组 [时间戳, 值])
        output_folder: 输出文件夹
        title_prefix: 图表标题前缀
        batch3_data_folder_path：ERA5文件夹，用于图表标题命名
        ylabel: Y轴标签
        plot_common_period_only: 是否仅绘制公共时段的图像
        forward_hours: 预测数据较实测数据的提前小时数
        time_interval_hours: 绘图时X轴的间隔（小时）
        rotation_user: 绘图时X轴上时间的旋转角度
        print_on：执行函数时，是否打印信息
    返回:
        None (会为每个站点单独保存图像到指定目录)
    """

    # 0. 全局设置 Times New Roman 字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'

    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)

    # 创建子文件夹用于存储绘图
    plot_subfolder = "plots"  # 子文件夹名称
    plot_output_folder = os.path.join(output_folder, plot_subfolder)
    os.makedirs(plot_output_folder, exist_ok=True)

    # 获取文件夹中的所有文件名（去除后缀），并按升序排序
    file_names = []
    if os.path.isdir(batch3_data_folder_path):
        for filename in os.listdir(batch3_data_folder_path):
            if os.path.isfile(os.path.join(batch3_data_folder_path, filename)):
                # 去除文件后缀
                base_name = os.path.splitext(filename)[0]
                file_names.append(base_name)

    # 按文件名升序排序
    file_names.sort()

    # 1. 检查点位数量是否一致
    if len(batch1_data) != len(batch2_data):
        print(f"Fv6_validate_and_plot-Error: Number of monitoring points mismatch! "
              f"Forecast has {len(batch1_data)} points, observation has {len(batch2_data)} points")
        return

    num_points = len(batch1_data)

    # 2. 为每个点位单独绘制图像并保存
    for point_idx in range(num_points):
        # 获取当前站点的预报和观测数据
        forecast_data = batch1_data[point_idx]
        observed_data = batch2_data[point_idx]

        # 获取时间戳和值
        forecast_timestamps = forecast_data[:, 0]
        forecast_values = forecast_data[:, 1]

        observed_timestamps = observed_data[:, 0]
        observed_values = observed_data[:, 1]

        # 确定时间范围
        forecast_start, forecast_end = np.min(forecast_timestamps), np.max(forecast_timestamps)
        observed_start, observed_end = np.min(observed_timestamps), np.max(observed_timestamps)

        # 3. 根据参数决定是否仅使用公共时段数据
        if plot_common_period_only:
            # 3.1 计算公共时间段的开始和结束（考虑预测数据提前显示）
            common_start_orig = max(forecast_start, observed_start)
            common_end = min(forecast_end, observed_end)
            # common_start = common_start_orig - forward_hours * 3600     # 显示预测值提前
            common_start = common_start_orig    # 显示公共区域

            if common_start > common_end:
                print(f"Fv6_validate_and_plot-Warning for site {point_idx + 1}: "
                      "No overlapping time period between forecast and observation data!")
                continue

            # 3.2 提取公共时间段的数据
            forecast_mask = (forecast_timestamps >= common_start) & (forecast_timestamps <= common_end)
            observed_mask = (observed_timestamps >= common_start) & (observed_timestamps <= common_end)

            forecast_to_plot = forecast_values[forecast_mask]
            forecast_ts_to_plot = forecast_timestamps[forecast_mask]
            observed_to_plot = observed_values[observed_mask]
            observed_ts_to_plot = observed_timestamps[observed_mask]

            # 时间范围标签
            time_range_str = (
                f"\n{datetime.fromtimestamp(common_start).strftime('%Y-%m-%d %H:%M')} - "
                f"{datetime.fromtimestamp(common_end).strftime('%Y-%m-%d %H:%M')}"
            )
        else:
            # 3.3 使用所有数据
            forecast_to_plot = forecast_values
            forecast_ts_to_plot = forecast_timestamps
            observed_to_plot = observed_values
            observed_ts_to_plot = observed_timestamps

            # # 检查数据范围是否一致
            # if forecast_start != observed_start or forecast_end != observed_end:
            #     print(f"Fv6_validate_and_plot-Warning for site {point_idx + 1}: Time range mismatch!")

            # 时间范围标签
            time_range_str = (
                f"\nFore: {datetime.fromtimestamp(forecast_start).strftime('%Y-%m-%d %H:%M')} - "
                f"{datetime.fromtimestamp(forecast_end).strftime('%Y-%m-%d %H:%M')}\n"
                f"Obs: {datetime.fromtimestamp(observed_start).strftime('%Y-%m-%d %H:%M')} - "
                f"{datetime.fromtimestamp(observed_end).strftime('%Y-%m-%d %H:%M')}"
            )

        # 4. 创建新的图形对象
        plt.figure(figsize=(15, 4))

        # 绘制预报数据
        plt.plot(forecast_ts_to_plot, forecast_to_plot,
                 label='Forecast', color='#f9c00c', alpha=0.5, linewidth=2)

        # 绘制观测数据
        plt.plot(observed_ts_to_plot, observed_to_plot,
                 label='Observed', color='#7200da', alpha=0.5, linewidth=2)

        # 美化图形
        # 获取当前站点的文件名（按升序排序）
        if point_idx < len(file_names):
            file_title = file_names[point_idx]
        else:
            # 如果站点数多于文件名数，循环使用文件名
            file_title = file_names[point_idx % len(file_names)]

        # 美化图形
        plt.title(f'{title_prefix}{file_title}')  # 图中标题以batch3_data_folder_path中各子文件名命名
        plt.ylabel(ylabel)
        plt.legend(loc='upper left')   # 统一设置图例位置（右上角）

        # 设置x轴刻度
        ax = plt.gca()

        # 使用实际数据的时间戳范围
        min_ts = min(np.min(forecast_ts_to_plot), np.min(observed_ts_to_plot))
        max_ts = max(np.max(forecast_ts_to_plot), np.max(observed_ts_to_plot))
        ax.set_xlim(min_ts, max_ts)     # 新添加：设置X轴精确范围，避免留白

        tick_locations, tick_labels = Fv6_generate_time_ticks(
            min_ts, max_ts, time_interval_hours)

        ax.set_xticks(tick_locations)
        ax.set_xticklabels(tick_labels, rotation=rotation_user)
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.xlabel('Time')
        plt.tight_layout()

        # 5. 为每个站点单独保存图像
        output_path = os.path.join(plot_output_folder, f'{file_title}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        if print_on:
            print(f"Site {file_title} plot saved to: {output_path}")