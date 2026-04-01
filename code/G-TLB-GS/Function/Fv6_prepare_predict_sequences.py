import numpy as np


def Fv6_prepare_predict_sequences(sequence_data, timestamps, history_steps, time_interval):
    """
    为单个站点准备预测序列数据

    参数:
        sequence_data (np.ndarray): 单站点的序列数据 [时间步数, 特征数]
        timestamps (np.ndarray): 单站点的时间戳 [时间步数]
        history_steps (float): 历史时间步数（1小时1步）
        time_interval (int): 时间间隔（秒）

    返回:
        tuple: (X_seq, pred_ts)
        X_seq: 三维输入序列 [样本数, 历史长度, 特征数]
        pred_ts: 预测时间点的时间戳 [样本数]
    """
    # 计算历史数据长度
    history_length = int(1 + history_steps * 3600 / time_interval)

    # 获取数据维度
    num_timesteps, num_features = sequence_data.shape

    # 验证输入序列长度是否有效
    if history_length > num_timesteps:
        raise ValueError(
            f"历史时间步长({history_steps}小时)需要的长度({history_length}) "
            f"大于实际时间步数({num_timesteps})"
        )

    # 计算可生成的样本数
    num_samples = num_timesteps - history_length + 1

    # 准备输入序列
    X_seq = np.zeros((num_samples, history_length, num_features))

    # 填充序列
    for i in range(num_samples):
        X_seq[i] = sequence_data[i:i + history_length]  # 由过去到当前

    # 预测时间点是每个输入序列的最后一个时间点（如果是有未来时间步长，则为未来时间步长的第一个时刻）
    pred_ts = timestamps[history_length - 1:history_length - 1 + num_samples]   # 这个切片操作会得到 num_samples 个元素

    return X_seq, pred_ts