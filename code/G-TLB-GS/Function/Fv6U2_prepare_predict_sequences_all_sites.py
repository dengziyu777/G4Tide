import numpy as np


def Fv6U2_prepare_predict_sequences(sequence_data, timestamps, history_steps, time_interval, future_steps):
    """
    为单个站点准备预测序列数据和未来时间戳矩阵

    参数:
        sequence_data (np.ndarray): 单站点的序列数据 [时间步数, 特征数]
        timestamps (np.ndarray): 单站点的时间戳 [时间步数]
        history_steps (float): 历史时间步长（小时）
        time_interval (int): 时间间隔（秒）
        future_steps (int): 预测的未来时间步数，包含当前时间步

    返回:
        tuple: (X_seq, pred_ts_matrix)
        X_seq: 三维输入序列 [样本数, 历史长度, 特征数]
        pred_ts_matrix: 未来时间戳矩阵 [样本数, future_steps]
    """
    # 计算历史数据长度（包含当前时间步）
    history_length = int(1 + history_steps * 3600 / time_interval)  # 历史时间步数量包含当前时刻

    # 获取数据维度
    num_timesteps, num_features = sequence_data.shape

    # 验证输入序列长度是否有效
    if history_length > num_timesteps:
        raise ValueError(
            f"历史时间步长({history_steps}小时)需要的长度({history_length}) "
            f"大于实际时间步数({num_timesteps})"
        )

    # 计算可生成的样本数 (保证有足够的未来数据)
    num_samples = num_timesteps - history_length - future_steps + 2     # 当历史未来均为1时，此处应等于时样本数即全部时刻数

    # 准备输入序列
    X_seq = np.zeros((num_samples, history_length, num_features))

    # 填充序列
    for i in range(num_samples):
        X_seq[i] = sequence_data[i:i + history_length]

    # 生成未来时间戳矩阵 [样本数, future_steps]
    pred_ts_matrix = np.zeros((num_samples, future_steps), dtype=timestamps.dtype)

    for i in range(num_samples):
        # 当前样本的历史结束时间点
        current_ts = timestamps[i + history_length - 1]     # 当前时刻

        # 计算从当前时间点开始的未来时间戳
        for j in range(future_steps):   # 如果 future_steps 的值为 5，那么循环变量 j 将依次取值为 0, 1, 2, 3, 4
            pred_ts_matrix[i, j] = current_ts + j * time_interval

    return X_seq, pred_ts_matrix



def Fv6U2_prepare_predict_sequences_all_sites(sequence_data_per_site, timestamps_per_site, site_params):
    """
    为所有站点准备预测序列数据和未来时间戳矩阵

    参数:
        sequence_data_per_site: 列表，每个元素是一个站点的序列数据 [时间步数, 特征数]
        timestamps_per_site: 列表，每个元素是一个站点的时间戳 [时间步数]
        site_params: 字典，站点索引到参数的映射

    返回:
        tuple: (site_sequences, predict_timestamps_matrix)
        site_sequences: 字典，站点索引到输入序列的映射 [样本数, 历史长度, 特征数]
        predict_timestamps_matrix: 字典，站点索引到未来时间戳矩阵的映射 [样本数, 未来长度]
    """
    site_sequences = {}
    predict_timestamps_matrix = {}

    for site_idx in range(len(sequence_data_per_site)):
        # 获取当前站点的数据和参数
        site_data = sequence_data_per_site[site_idx]
        site_ts = timestamps_per_site[site_idx]
        site_param = site_params[site_idx]

        # 计算未来步长 (基于use_backward_hours参数)
        future_steps = int(1 + site_param["use_backward_hours"] * 3600 / site_param["use_time_interval"])   # 未来时间步数量，包含当前时刻

        # 准备当前站点的预测序列
        X_seq, pred_ts_matrix = Fv6U2_prepare_predict_sequences(
            sequence_data=site_data,
            timestamps=site_ts,
            history_steps=site_param["use_forward_hours"],
            time_interval=site_param["use_time_interval"],
            future_steps=future_steps   # 未来时间步 数量？
        )

        # 存储结果
        site_sequences[site_idx] = X_seq
        predict_timestamps_matrix[site_idx] = pred_ts_matrix

    return site_sequences, predict_timestamps_matrix