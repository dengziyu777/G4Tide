

def Fv1_preprocess_data_LSTM(batch1_data, lookback, forecast_horizon, train_ratio=0.7, val_ratio=0.15,
                             standardize=False):
    """
    预处理实测数据，针对各个实测点位各个时段分别构建序列数据集（1个实测点位1个单独时段，后续称为1个站点）
    各站点使用自己的历史数据预测自己的未来值

    参数:
        batch1_data: 原始站点数据列表
        lookback: 历史事件步长
        forecast_horizon: 未来预测步长
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        standardize: 是否对数据进行标准化，默认为False

    返回:
        处理后的数据集字典，包含全部站点的合并数据
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import torch

    def create_sequences_for_site(site_data, lookback, forecast_horizon):
        """
        为单个站点创建输入输出序列
        """
        # 提取潮位数据（第二列）
        water_levels = site_data[:, 1]

        # 处理缺失值
        df = pd.DataFrame(water_levels, columns=['water_level'])
        df_filled = df.interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(
            method='ffill')
        measurement_data_clean = df_filled.values.flatten()  # 将DataFrame转换回NumPy数组并展平为一维数组；measurement_data_clean为完整的数据

        # 创建序列
        X, y = [], []
        n_samples = len(measurement_data_clean) - lookback - forecast_horizon + 1  # 样本数

        for i in range(n_samples):
            # 输入：过去lookback个时间步的数据
            X.append(measurement_data_clean[i:(i + lookback)])
            # 输出：未来forecast_horizon个时间步的数据
            y.append(measurement_data_clean[i + lookback:i + lookback + forecast_horizon])

        return np.array(X), np.array(y), measurement_data_clean

    print("开始预处理实测数据...")
    print(f"参数: 历史时间步数={lookback}, 未来时间步数={forecast_horizon}, 是否标准化={standardize}")

    # 1. 检查数据有效性
    if not batch1_data or len(batch1_data) == 0:
        raise ValueError("输入数据为空")

    # 2. 为每个站点单独创建序列
    all_X = []
    all_y = []
    all_scalers = []
    site_info = []

    for site_idx, site_data in enumerate(batch1_data):
        print(f"处理站点 {site_idx + 1}/{len(batch1_data)}...")  # 站点编号从1开始

        # 检查数据长度是否足够
        if len(site_data) < lookback + forecast_horizon:
            print(f"警告: 站点{site_idx + 1}数据长度不足，已跳过")
            continue

        # 创建该站点的序列
        X_site, y_site, measurement_data_clean = create_sequences_for_site(
            site_data, lookback, forecast_horizon
        )

        if len(X_site) == 0:
            print(f"警告: 站点{site_idx + 1}无法生成有效序列，已跳过")
            continue

        print(f"  站点{site_idx + 1}生成 {len(X_site)} 个序列")

        # 根据standardize参数决定是否进行标准化
        if standardize:
            # 为该站点数据创建标准化器
            scaler = StandardScaler()  # Mean-Variance Standardization（均值-方差标准化）
            measurement_data_reshaped = measurement_data_clean.reshape(-1, 1)
            scaler.fit(measurement_data_reshaped)

            # 标准化序列数据
            X_site_reshaped = X_site.reshape(-1, 1)
            X_site_scaled = scaler.transform(X_site_reshaped).reshape(X_site.shape)

            y_site_reshaped = y_site.reshape(-1, 1)
            y_site_scaled = scaler.transform(y_site_reshaped).reshape(y_site.shape)

            # 添加到总数据集
            all_X.append(X_site_scaled)
            all_y.append(y_site_scaled)
        else:
            # 不进行标准化，使用原始数据
            scaler = None
            all_X.append(X_site)
            all_y.append(y_site)

        all_scalers.append(scaler)
        site_info.append({
            'site_index': site_idx,
            'original_length': len(site_data),
            'sequence_count': len(X_site),
            'scaler': scaler
        })

    if not all_X:
        raise ValueError("没有生成任何有效序列")

    # 3. 合并所有站点的序列
    X_combined = np.vstack(all_X)
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
    X_train = torch.FloatTensor(X_train)  # 将NumPy数组转换为PyTorch张量
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # 调整维度以适应"LSTM"输入 (batch_size, seq_len, 1)
    X_train = X_train.unsqueeze(-1)  # 使用unsqueeze(-1)在最后添加一个维度
    X_val = X_val.unsqueeze(-1)
    X_test = X_test.unsqueeze(-1)

    # 如果forecast_horizon>1，确保y的维度正确
    if forecast_horizon > 1 and len(y_train.shape) == 1:  # 是多步预测andy的维度不正确 (只有1维，应该是2维)
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
        'standardize': standardize
    }

    print("数据预处理完成!")
    return result