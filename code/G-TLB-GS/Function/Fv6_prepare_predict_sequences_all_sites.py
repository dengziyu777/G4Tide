from source.Fv6_prepare_predict_sequences import Fv6_prepare_predict_sequences

def Fv6_prepare_predict_sequences_all_sites(sequence_data_per_site, timestamps_per_site, site_params):
    """
    为所有站点准备预测序列数据

    参数:
        sequence_data_per_site: 列表，每个元素是一个站点的序列数据 [时间步数, 特征数]
        timestamps_per_site: 列表，每个元素是一个站点的时间戳 [时间步数]
        site_params: 字典，站点索引到参数的映射

    返回:
        tuple: (site_sequences, predict_timestamps)
        site_sequences: 字典，站点索引到输入序列的映射 [样本数, 历史长度, 特征数]
        predict_timestamps: 字典，站点索引到预测时间戳的映射 [样本数]（如果是有未来时间步长，则为未来时间步长的第一个时刻）
    """
    site_sequences = {}
    predict_timestamps = {}

    for site_idx in range(len(sequence_data_per_site)):
        # 获取当前站点的数据和参数
        site_data = sequence_data_per_site[site_idx]
        site_ts = timestamps_per_site[site_idx]
        site_param = site_params[site_idx]

        # 准备当前站点的预测序列
        X_seq, pred_ts = Fv6_prepare_predict_sequences(
            sequence_data=site_data,
            timestamps=site_ts,
            history_steps=site_param["use_forward_hours"],  # 历史时间步数（1h1步）
            time_interval=site_param["use_time_interval"]
        )

        # 存储结果
        site_sequences[site_idx] = X_seq
        predict_timestamps[site_idx] = pred_ts

        # print(f" 站点 {site_idx} 准备完成: 输入序列形状={X_seq.shape}, 预测时间点数={len(pred_ts)}")

    return site_sequences, predict_timestamps