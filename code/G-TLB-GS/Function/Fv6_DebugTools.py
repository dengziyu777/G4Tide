import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

def debug_print_meteo_data(timestamps_meteo, meteo_data, num_samples=3, num_sites_to_show=2):
    """
    打印气象数据的调试信息

    参数:
        timestamps_meteo: 时间戳列表 (datetime对象列表)
        meteo_data: 气象数据三维数组 (形状为[时间步, 站点, 特征])
        num_samples: 要显示的时间步数量 (默认3)
        num_sites_to_show: 每个时间步要显示的站点数量 (默认2)
    """
    # 检查是否有数据
    if meteo_data.size == 0:
        print("气象数据为空！")
        return

    total_timesteps, num_stations, num_features = meteo_data.shape

    # 打印基本信息
    print(f"\n气象数据基本信息:")
    print(f"时间戳数量: {total_timesteps}")
    print(f"气象数据数组形状: {meteo_data.shape} (时间步×站点数×特征数)")
    print(f"起始时间: {timestamps_meteo[0]}")
    print(f"结束时间: {timestamps_meteo[-1]}")
    print(f"站点数量: {num_stations}")
    print(f"每个站点的特征数量: {num_features}")

    # 打印样本数据
    print(f"\n前{num_samples}个时间步的气象数据样例:")
    for i in range(min(num_samples, total_timesteps)):
        if i < len(timestamps_meteo):
            time_str = timestamps_meteo[i].strftime("%Y-%m-%d %H:%M:%S")
        else:
            time_str = "未知时间"

        print(f"时间步 {i} ({time_str}):")

        # 仅显示指定数量的站点
        for site in range(min(num_sites_to_show, num_stations)):
            features = meteo_data[i, site, :]

            # 创建格式化的特征值列表
            features_str = []
            for j, value in enumerate(features):
                # 使用科学记数法表示极小或极大的值
                if abs(value) > 1000 or (abs(value) < 0.01 and value != 0):
                    formatted_value = f"{value:.4e}"
                else:
                    formatted_value = f"{value:.4f}"

                # 添加特征索引编号
                features_str.append(f"F#{j + 1}: {formatted_value}")

            # 打印格式化后的特征值
            print(f"  站点 {site + 1}: [{', '.join(features_str)}]")

    # print(f"\n气象数据范围:")
    # print(f"全局最小值: {np.min(meteo_data):.6f}")
    # print(f"全局最大值: {np.max(meteo_data):.6f}")
    # print(f"全局平均值: {np.mean(meteo_data):.6f}")
    # print(f"特征平均值: {np.mean(meteo_data, axis=(0, 1))}")


def debug_plot_interpolated_scatter(time_data, forecast_data, obs_data, save_path, station_idx=0):
    """
    绘制插值后的预报数据与观测数据的二维散点图（仅使用公共时间段数据）- 单站点版本

    参数:
        time_data (np.array): 公共时间段的时间戳数组(UNIX时间戳)
        forecast_data (np.array): 公共时间段的预报数据(一维数组)
        obs_data (np.array): 公共时间段的观测数据(一维数组)
        save_path (str): 图片保存路径
        station_idx (int): 站点索引
    """
    plt.figure(figsize=(15, 8))

    # 转换时间戳为datetime对象
    time_dt = [datetime.fromtimestamp(t) for t in time_data]

    # 确保数据长度一致
    assert len(forecast_data) == len(obs_data) == len(time_dt), "数据长度不一致"

    # 绘制散点图
    plt.scatter(time_dt, forecast_data, c='blue', alpha=0.7,
                label=f'Forecast (Station {station_idx + 1})', marker='o')
    plt.scatter(time_dt, obs_data, c='red', alpha=0.7,
                label=f'Observation (Station {station_idx + 1})', marker='x')

    # 添加误差连接线
    for t, f, o in zip(time_dt, forecast_data, obs_data):
        plt.plot([t, t], [f, o], 'gray', linestyle=':', alpha=0.2)

    # 计算并标注误差指标
    mae = mean_absolute_error(obs_data, forecast_data)
    rmse = np.sqrt(mean_squared_error(obs_data, forecast_data))
    r2 = r2_score(obs_data, forecast_data)

    stats_text = f"MAE: {mae:.3f} m\nRMSE: {rmse:.3f} m\nR²: {r2:.3f}"
    plt.annotate(stats_text, xy=(0.02, 0.90), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', alpha=0.2))

    # 设置图表属性
    plt.title(f"Tide Level: Forecast vs Observation (Station {station_idx + 1})")
    plt.xlabel("Time")
    plt.ylabel("EL(m)")
    plt.legend()
    plt.grid(True)

    # 优化日期显示
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()