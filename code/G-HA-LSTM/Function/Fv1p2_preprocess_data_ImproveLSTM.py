
def Fv1_preprocess_data_ImproveLSTM(batch1_data, batch2_data, lookback, forecast_horizon, train_ratio=0.7,
                                    val_ratio=0.15, standardize=False):
    """
    预处理数据，构建用于LSTM训练的数据集

    输入X:
        - batch1_data的[i:(i + lookback)] (历史实测数据)
        - batch2_data的[i + lookback:i + lookback + forecast_horizon] (未来HA数据)
    输出Y:
        - batch1_data的[i + lookback:i + lookback + forecast_horizon] (未来实测数据)

    参数:
        batch1_data: 实测数据列表
        batch2_data: HA数据列表
        lookback: 历史时间步长
        forecast_horizon: 未来预测步长
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        standardize: 是否对数据进行标准化，默认为False

    返回:
        处理后的数据集字典
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import torch

    def create_sequences_for_site(measurement_data, ha_data, lookback, forecast_horizon):
        """
        为单个站点创建输入输出序列
        输入X: [历史实测数据, 未来HA数据]
        输出Y: 未来实测数据
        """
        # 提取实测潮位数据
        measurement_levels = measurement_data[:, 1]
        ha_levels = ha_data[:, 1]

        # 处理缺失值
        measurement_df = pd.DataFrame(measurement_levels, columns=['measurement'])
        measurement_df_filled = measurement_df.interpolate(method='linear', limit_direction='both').fillna(
            method='bfill').fillna(method='ffill')
        measurement_data_clean = measurement_df_filled.values.flatten()

        ha_df = pd.DataFrame(ha_levels, columns=['ha'])
        ha_df_filled = ha_df.interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(
            method='ffill')
        ha_data_clean = ha_df_filled.values.flatten()

        # 确保两个数据源长度相同
        min_length = min(len(measurement_data_clean), len(ha_data_clean))
        measurement_data_clean = measurement_data_clean[:min_length]
        ha_data_clean = ha_data_clean[:min_length]

        # 创建序列
        X_measurement, X_ha, X_combined_list, y = [], [], [], []
        n_samples = len(measurement_data_clean) - lookback - forecast_horizon + 1

        for i in range(n_samples):
            # 输入X的第一部分：batch1_data的[i:(i + lookback)] (历史实测数据)
            hist_measurement = measurement_data_clean[i:(i + lookback)]

            # 输入X的第二部分：batch2_data的[i + lookback:i + lookback + forecast_horizon] (未来HA数据)
            future_ha = ha_data_clean[i + lookback:i + lookback + forecast_horizon]

            # 组合输入X: 历史实测 + 未来HA
            X_combined = np.concatenate([hist_measurement, future_ha])

            # 输出Y: batch1_data的[i + lookback:i + lookback + forecast_horizon] (未来实测数据)
            future_measurement = measurement_data_clean[i + lookback:i + lookback + forecast_horizon]

            X_measurement.append(hist_measurement)
            X_ha.append(future_ha)
            X_combined_list.append(X_combined)  # 使用不同的变量名
            y.append(future_measurement)

        return np.array(X_measurement), np.array(X_ha), np.array(X_combined_list), np.array(
            y), measurement_data_clean, ha_data_clean

    print("开始预处理数据...")
    print(f"参数: 历史时间步数={lookback}, 未来时间步数={forecast_horizon}, 是否标准化={standardize}")

    # 1. 检查数据有效性
    if not batch1_data or len(batch1_data) == 0:
        raise ValueError("实测数据为空")

    if not batch2_data or len(batch2_data) == 0:
        raise ValueError("HA数据为空")

    if len(batch1_data) != len(batch2_data):
        raise ValueError(f"实测数据站点数({len(batch1_data)})与HA数据站点数({len(batch2_data)})不一致")

    # 2. 为每个站点单独创建序列
    all_X_measurement = []
    all_X_ha = []
    all_X_combined = []
    all_y = []
    all_scalers_measurement = []
    all_scalers_ha = []
    site_info = []

    for site_idx in range(len(batch1_data)):
        print(f"处理站点 {site_idx + 1}/{len(batch1_data)}...")

        measurement_data = batch1_data[site_idx]
        ha_data = batch2_data[site_idx]

        # 检查数据长度是否足够
        if len(measurement_data) < lookback + forecast_horizon:
            print(f"警告: 站点{site_idx + 1}实测数据长度不足，已跳过")
            continue

        if len(ha_data) < lookback + forecast_horizon:
            print(f"警告: 站点{site_idx + 1}HA数据长度不足，已跳过")
            continue

        # 创建该站点的序列
        X_measurement_site, X_ha_site, X_combined_site, y_site, measurement_data_clean, ha_data_clean = create_sequences_for_site(
            measurement_data, ha_data, lookback, forecast_horizon
        )

        if len(X_combined_site) == 0:
            print(f"警告: 站点{site_idx + 1}无法生成有效序列，已跳过")
            continue

        print(f"  站点{site_idx + 1}生成 {len(X_combined_site)} 个序列")

        # 根据standardize参数决定是否进行标准化
        if standardize:
            # 为实测数据创建标准化器
            scaler_measurement = StandardScaler()
            measurement_data_reshaped = measurement_data_clean.reshape(-1, 1)
            scaler_measurement.fit(measurement_data_reshaped)

            # 为HA数据创建标准化器
            scaler_ha = StandardScaler()
            ha_data_reshaped = ha_data_clean.reshape(-1, 1)
            scaler_ha.fit(ha_data_reshaped)

            # 标准化序列数据
            # 标准化实测历史部分
            X_measurement_reshaped = X_measurement_site.reshape(-1, 1)
            X_measurement_scaled = scaler_measurement.transform(X_measurement_reshaped).reshape(
                X_measurement_site.shape)

            # 标准化HA未来部分
            X_ha_reshaped = X_ha_site.reshape(-1, 1)
            X_ha_scaled = scaler_ha.transform(X_ha_reshaped).reshape(X_ha_site.shape)

            # 组合标准化后的输入
            X_combined_scaled = np.concatenate([X_measurement_scaled, X_ha_scaled], axis=1)

            # 标准化输出
            y_reshaped = y_site.reshape(-1, 1)
            y_scaled = scaler_measurement.transform(y_reshaped).reshape(y_site.shape)

            # 使用标准化后的数据
            X_measurement_to_add = X_measurement_scaled
            X_ha_to_add = X_ha_scaled
            X_combined_to_add = X_combined_scaled
            y_to_add = y_scaled
        else:
            # 不进行标准化，使用原始数据
            scaler_measurement = None
            scaler_ha = None
            # 组合输入
            X_combined_to_add = np.concatenate([X_measurement_site, X_ha_site], axis=1)
            # 使用原始数据
            X_measurement_to_add = X_measurement_site
            X_ha_to_add = X_ha_site
            y_to_add = y_site

        # 添加到总数据集
        all_X_measurement.append(X_measurement_to_add)
        all_X_ha.append(X_ha_to_add)
        all_X_combined.append(X_combined_to_add)
        all_y.append(y_to_add)
        all_scalers_measurement.append(scaler_measurement)
        all_scalers_ha.append(scaler_ha)
        site_info.append({
            'site_index': site_idx,
            'measurement_length': len(measurement_data),
            'ha_length': len(ha_data),
            'sequence_count': len(X_combined_site),
            'scaler_measurement': scaler_measurement,
            'scaler_ha': scaler_ha
        })

    if not all_X_combined:
        raise ValueError("没有生成任何有效序列")

    # 3. 合并所有站点的序列
    X_combined = np.vstack(all_X_combined)
    y_combined = np.vstack(all_y)

    print(f"合并后的数据集形状: X={X_combined.shape}, y={y_combined.shape}")
    print(f"总共从 {len(site_info)} 个站点生成 {len(X_combined)} 个序列")

    # 4. 数据集划分
    n_total = len(X_combined)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    # 随机打乱数据
    indices = np.random.permutation(n_total)
    X_shuffled = X_combined[indices]
    y_shuffled = y_combined[indices]

    # 划分数据集
    X_train, y_train = X_shuffled[:n_train], y_shuffled[:n_train]
    X_val, y_val = X_shuffled[n_train:n_train + n_val], y_shuffled[n_train:n_train + n_val]
    X_test, y_test = X_shuffled[n_train + n_val:], y_shuffled[n_train + n_val:]

    print(f"数据集划分: 训练集{n_train}, 验证集{n_val}, 测试集{n_test}")

    # 5. 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # 调整维度以适应LSTM输入 (batch_size, seq_len, 1)
    # 输入X的形状是 (batch_size, lookback+forecast_horizon)，需要变为 (batch_size, lookback+forecast_horizon, 1)
    X_train = X_train.unsqueeze(-1)
    X_val = X_val.unsqueeze(-1)
    X_test = X_test.unsqueeze(-1)

    # 如果forecast_horizon>1，确保y的维度正确
    if forecast_horizon > 1 and len(y_train.shape) == 1:
        y_train = y_train.unsqueeze(-1)
        y_val = y_val.unsqueeze(-1)
        y_test = y_test.unsqueeze(-1)

    print(f"最终数据形状:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # 返回结果
    result = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'site_info': site_info,
        'lookback': lookback,
        'forecast': forecast_horizon,
        'all_scalers_measurement': all_scalers_measurement,
        'all_scalers_ha': all_scalers_ha,
        'standardize': standardize
    }

    print("数据预处理完成!")
    return result