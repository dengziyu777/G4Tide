import numpy as np


def Fv6_adaptive_smoothing(target_series, reference_series, batch2_data_time_interval_adjust):
    """
    依据参考序列平滑度自适应平滑目标序列
    :param target_series: 待平滑序列（数值列表或一维数组）
    :param reference_series: 参考序列（数值列表或一维数组）
    batch2_data_time_interval_adjust：DL模型预测数据时间间隔
    :return: 平滑后的目标序列
    """
    # 计算参考序列平滑度指标（一阶差分标准差）
    ref_diff = np.diff(reference_series)    # 差分（相邻数据的差值）的标准差越小，说明序列变化越平缓，越平滑
    if np.std(ref_diff) > 0:
        ref_smoothness = 1 / np.std(ref_diff)  # 平滑度与差分标准差成反比
    else:
        # 如果参考序列完全平滑（差分标准差=0），使用安全值
        ref_smoothness = len(target_series) / 5

    # 计算目标序列原始波动度
    target_diff = np.diff(target_series)
    target_volatility = np.std(target_diff) if np.std(target_diff) > 0 else 1.0 # 如果差分标准差=0（完全无波动），默认赋值为1.0

    # 计算动态窗口大小（关键公式）；参考序列越平滑 → 窗口越大，目标序列噪声越大 → 窗口越大
    max_hours = 6  # 最大平滑时间窗口（小时）
    max_data_points = max_hours * 3600 // batch2_data_time_interval_adjust  # 转换为数据点数

    base_window = max(3, int(ref_smoothness * len(target_series) / (target_volatility + 1e-5)))
    window_size = min(base_window, len(target_series) // 2, max_data_points)

    # 边界镜像填充（处理边缘效应）
    padding = window_size // 2  # 在数据两端复制镜像值，避免边界处窗口不完整导致的失真
    padded = np.pad(target_series, (padding, padding), 'reflect')   # 原始 target_series 被镜像填充（reflect模式）后生成 padded

    # 执行移动平均
    smoothed = np.zeros_like(target_series)
    for i in range(len(target_series)):
        start_idx = i
        end_idx = i + window_size
        smoothed[i] = np.mean(padded[start_idx:end_idx])

    return smoothed, window_size