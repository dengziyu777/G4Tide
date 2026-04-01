import numpy as np
from scipy.interpolate import CubicSpline


def Fv6_adjust_observation_interval(time_data, tide_data, original_interval, target_interval):
    """
    调整观测数据的时间间隔（通过三次样条插值）

    改进点：
    1. 更精确地生成目标时间序列（避免超出范围）
    2. 添加边界条件处理（减少端点振荡）
    3. 验证输入数据有效性
    4. 处理单站点的特殊情况

    参数:
        time_data (np.array): 严格递增的UNIX时间戳数组
        tide_data (np.array): 潮位数据(2D数组，每列代表一个站点)
        original_interval (int): 原始数据时间间隔(秒)
        target_interval (int): 目标时间间隔(秒)

    返回:
        tuple: (调整后的时间戳数组, 插值后的潮位数据)
    """
    # 验证输入数据
    if target_interval <= 0:
        raise ValueError("目标间隔必须是正数 (输入: {})".format(target_interval))

    if not np.all(np.diff(time_data) > 0):
        raise ValueError("时间数据必须严格递增")

    if len(time_data) < 3:
        raise ValueError("至少需要3个数据点进行三次样条插值")

    # 生成新的时间序列 (精确控制边界)
    start_time = time_data[0]
    end_time = time_data[-1]

    # 计算需要的时间点数量 (确保覆盖整个范围)
    total_seconds = end_time - start_time
    num_points = int(total_seconds / target_interval) + 1
    new_times = start_time + np.arange(num_points) * target_interval

    # 处理可能的小数点误差，确保不超出原范围
    new_times = np.minimum(new_times, end_time)

    # 检查插值需求
    ratio = target_interval / original_interval
    if not (ratio.is_integer() or (1 / ratio).is_integer()):
        print(f"警告: 从{original_interval}秒插值到{target_interval}秒可能导致精度损失")

    # 对每个站点进行插值
    if tide_data.ndim == 1:     # NumPy数组有一个属性ndim，它表示数组的维数
        tide_data = tide_data.reshape(-1, 1)  # 确保为二维

    interpolated_tides = []
    for col in range(tide_data.shape[1]):
        values = tide_data[:, col]

        # 使用natural cubic spline (端点二阶导数为零)
        cs = CubicSpline(time_data, values, bc_type='natural')
        station_data = cs(new_times)
        interpolated_tides.append(station_data)

    interpolated_tides = np.column_stack(interpolated_tides)


    return new_times, interpolated_tides