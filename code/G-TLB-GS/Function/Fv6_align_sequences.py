import numpy as np


def Fv6_align_sequences(t1, d1, forward_hours_4train, t2, d2, time_interval):
    """
    通过线性插值对齐两个时间序列（将预报数据插值到观测数据的时间点）- 单站点版本

    参数:
        t1 (np.array): 预报数据的时间戳数组 (一维)
        d1 (np.array): 预报数据值数组 (一维)
        forward_hours_4train (int): 预报数据的提前小时数
        t2 (np.array): 观测数据的时间戳数组 (一维)
        d2 (np.array): 观测数据值数组 (一维)
        time_interval (int): 观测数据的时间间隔（秒）

    返回:
        tuple: (插值后的预报数据, 完整时间序列, 插值后的观测数据, 公共时间段的时间序列)
    """
    # 确保输入是一维数组
    t1 = np.asarray(t1).flatten()
    d1 = np.asarray(d1).flatten()
    t2 = np.asarray(t2).flatten()
    d2 = np.asarray(d2).flatten()

    # 1. 确定对齐的时间边界
    min_time_orig = max(t1.min(), t2.min())  # 实际对齐起点（观测数据起始）
    max_time = min(t1.max(), t2.max())  # 结束时间

    # 没有重叠时间段
    if min_time_orig > max_time:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # 2. 计算预报所需的提前量（整数倍时间间隔）
    forward_seconds = forward_hours_4train * 3600
    num_intervals = int(np.ceil(forward_seconds / time_interval))   # 考虑历史时间步后，应向前考虑多少数据
    effective_forward = num_intervals * time_interval

    # 3. 生成基准时间序列起点（考虑预测数据提前）
    min_time_base_source = min_time_orig - effective_forward
    if min_time_base_source >= t1.min():    # 若观测数据包含最早历史时刻，则从最早历史时刻考虑
        min_time_base = min_time_base_source
    else:       # 否则，就从预报、观测数据公共时刻考虑即可
        min_time_base = t1.min()

    # 4. 用相同基准生成完整时间序列
    t_target = np.arange(min_time_base, max_time, time_interval)  # 完整序列

    # 5. 定位重叠区域：后段就是观测数据对齐序列
    start_index = num_intervals  # 观测数据在完整序列中的起始索引，Python中索引从0开始
    t_target_orig = t_target[start_index:]  # 公共时间段的时间序列

    # 6. 插值预报数据（插值到完整序列）
    sorted_idx = np.argsort(t1)
    t1_sorted = t1[sorted_idx]
    d1_sorted = d1[sorted_idx]

    forecast_interp = np.interp(
        t_target,
        t1_sorted,
        d1_sorted,
        left=d1_sorted[0],
        right=d1_sorted[-1]
    )

    # 7. 插值观测数据（只插值到重叠区域）
    sorted_idx_obs = np.argsort(t2)
    t2_sorted = t2[sorted_idx_obs]
    d2_sorted = d2[sorted_idx_obs]

    obs_interp = np.interp(
        t_target_orig,
        t2_sorted,
        d2_sorted,
        left=d2_sorted[0],
        right=d2_sorted[-1]
    )

    return forecast_interp, t_target, obs_interp, t_target_orig