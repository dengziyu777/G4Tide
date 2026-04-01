import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import CubicSpline
import os
from joblib import dump
from datetime import datetime


def Fv6_prepare_sequence_data_with_meteo(
        X_tide_list,  # 预报潮位数据列表 [站点数]，每个元素是一维数组
        X_timestamps_list,  # 预测潮位时间戳列表 [站点数]，每个元素是一维数组
        Y_tide_list,  # 实测潮位数据列表 [站点数]，每个元素是一维数组
        Y_timestamps_list,  # 实测潮位时间戳列表 [站点数]，每个元素是一维数组
        batch3_data,  # 气象数据列表，每个元素是一个站点的数据数组 [时间戳, 特征值]
        time_interval_adjust,  # 时间步长(s)（潮位数据已插值至此步长）
        forward_hours,  # 历史时间长度(h)
        backward_hours,  # 预测未来长度(h)
        train_ratio,  # 训练集比例
        val_ratio,  # 验证集比例
        model_save_path,  # 保存标准化器的文件夹路径
        model_type,  # 模型类型
        case_name,  # 案例名称
        y_scaling_method,  # Y标准化方法: 'global'或'forecast_based'
        print_on = True
):
    """
    准备包含历史信息和气象数据的时序数据集 - 支持多站点独立处理

    参数:
        X_tide_list: 预报潮位数据列表 [站点数]，每个元素是一维数组
        X_timestamps_list: 预测潮位时间戳列表 [站点数]，每个元素是一维数组
        Y_tide_list: 实测潮位数据列表 [站点数]，每个元素是一维数组
        Y_timestamps_list: 实测潮位时间戳列表 [站点数]，每个元素是一维数组
        batch3_data: 气象数据列表，每个元素是一个站点的数据数组 [时间戳, 特征值]
        time_interval_adjust: 时间步长(s)（潮位数据已插值至此步长）
        forward_hours: 历史时间长度(h)
        backward_hours: 预测未来长度(h)
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        model_save_path: 保存标准化器的文件夹路径
        model_type: 模型类型
        case_name: 案例名称
        y_scaling_method: Y标准化方法
            'global' - 全局标准化
            'forecast_based' - 使用预报潮位的标准化参数
        print_on：执行函数时，是否打印信息

    返回:
        sequence_data: 时序数据字典
    """
    # 计算时间步长
    forward_length = int(1 + forward_hours * 3600 / time_interval_adjust)   # 输入层中需考虑时序数量；int向上取整；math.floor()向下取整；+1考虑一下当前步
    backward_length = int(1 + backward_hours * 3600 / time_interval_adjust) # 输出层中需考虑时序数量

    num_sites = len(X_tide_list)    # 列表中元素数量即为站点数量
    if num_sites == 0:
        raise ValueError("没有站点数据可处理")

    # 检查气象数据站点数量是否匹配
    if len(batch3_data) != num_sites:
        print(f"警告: 气象数据站点数量({len(batch3_data)})与潮位数据站点数量({num_sites})不匹配")
        num_sites = min(len(batch3_data), num_sites)
        print(f"将处理前 {num_sites} 个站点")

    # 为每个站点处理数据
    site_datasets = []  # 存储每个站点的数据集
    scaler_X_list = []  # 存储每个站点的输入标准化器
    all_Y_list = []  # 收集所有站点的训练输出数据
    valid_sites = 0  # 有效站点计数器

    for site in range(num_sites):
        if print_on:
            print(f"\n{'=' * 40} 站点 {site + 1} 时序准备开始 {'=' * 40}")

        # 获取当前站点的数据
        X_tide = X_tide_list[site]  # 预报潮位数据
        X_timestamps = X_timestamps_list[site]  # 预报潮位时间戳
        Y_tide = Y_tide_list[site]  # 观测潮位数据
        Y_timestamps = Y_timestamps_list[site]

        # 数据验证
        if len(X_tide) == 0 or len(Y_tide) == 0:
            print(f"站点 {site + 1} 潮位数据为空，跳过")
            continue

        # 获取当前站点的气象数据
        if site >= len(batch3_data):
            print(f"站点 {site + 1} 无气象数据，跳过")
            continue

        meteo_site_data = batch3_data[site]
        if len(meteo_site_data) == 0:
            print(f"站点 {site + 1} 气象数据为空，跳过")
            continue

        # 提取气象数据的时间戳和特征值
        meteo_timestamps = meteo_site_data[:, 0]    # 气象数据时间戳
        meteo_features = meteo_site_data[:, 1:]     # 气象数据
        num_meteo_features = meteo_features.shape[1]    # 气象特征数

        # 计算公共时间段起始点索引（在X_timestamps中）
        common_start_index = len(X_timestamps) - len(Y_timestamps)  # 结合Fv6_align_sequences，X_timestamps要么比Y_timestamps长(历史时间步长)，要么一样长
        if print_on:
            print(f"站点 {site + 1} 完整//公共时间段（同时间间隔）数据长度: {len(X_tide)} // {len(Y_tide)}")

        # 确定预报数据中可用的样本范围（需考虑向前历史时间步长、未来时间步长）
        start_idx = max(common_start_index,0)
        end_idx = len(X_timestamps)  # 终止索引为预报数据最后

        if start_idx >= end_idx:
            if print_on:
                print(f"站点 {site + 1} 没有足够的样本可用 (start={start_idx}, end={end_idx})")
            continue

        num_sequences = end_idx - start_idx
        if print_on:
            print(f"站点 {site + 1} 可用样本数量: {num_sequences}")

        # 创建当前站点的气象数据插值器
        interp_meteo_site = np.zeros((len(X_timestamps), num_meteo_features))

        # 为每个气象特征创建插值器
        for feature_idx in range(num_meteo_features):
            cs = CubicSpline(meteo_timestamps, meteo_features[:, feature_idx])  # 三次样条插值，将气象数据从原始的时间点插值到预报数据的时间点（以batch2_data_time_interval_adjust插值后）
            interp_meteo_site[:, feature_idx] = cs(X_timestamps)

        # 创建组合特征矩阵：[预报潮位, 气象特征1, 气象特征2, ...]
        combined_features = np.zeros((len(X_timestamps), 1 + num_meteo_features))
        combined_features[:, 0] = X_tide
        combined_features[:, 1:] = interp_meteo_site

        # 初始化序列容器
        X_site = np.zeros((num_sequences, forward_length, 1 + num_meteo_features))
        Y_site = np.zeros((num_sequences, backward_length))

        # 填充每个样本
        for i in range(num_sequences):
            seq_start = i   # 结合Fv6_align_sequences，预报数据要么比观测数据长(历史时间步)，要么一样长
            seq_end = seq_start + forward_length    # forward_length在本函数稍前向上取整

            # 输入序列: [seq_start, seq_end)
            X_site[i] = combined_features[seq_start:seq_end, :]     # 由forward_hours到当前，预测数据+气象数据

            # 目标序列: [seq_end, seq_end + backward_length)
            # 注意: 序列末尾对应的实测数据索引
            target_idx = (seq_end - 1) - common_start_index     # (seq_end - 1)为当前在预报数据的位置，- common_start_index，为当前在实测数据中位置
            if target_idx < 0 or target_idx + backward_length > len(Y_tide):
                # 使用 -9999 表示无效数据
                Y_site[i] = np.full(backward_length, -9999)
            else:
                Y_site[i] = Y_tide[target_idx:target_idx + backward_length]     # 由当前到backwards

        # 过滤掉目标包含无效值的序列
        valid_indices = [i for i in range(num_sequences) if not np.any(Y_site[i] == -9999)]

        if not valid_indices:
            if print_on:
                print(f"站点 {site + 1} 没有有效序列，跳过")
            continue

        num_valid_sequences = len(valid_indices)
        X_site = X_site[valid_indices]
        Y_site = Y_site[valid_indices]
        if print_on:
            print(f"站点 {site + 1} 有效序列数量: ({num_valid_sequences}/{num_sequences})")   # -9999的是无效数据

        # 数据集划分
        indices = np.arange(num_valid_sequences)
        np.random.shuffle(indices)

        train_size = int(train_ratio * num_valid_sequences)
        val_size = int(val_ratio * num_valid_sequences)
        test_size = num_valid_sequences - train_size - val_size

        if train_size <= 0 or val_size <= 0 or test_size <= 0:
            if print_on:
                print(f"站点 {site + 1} 的训练//验证//测试集中数据数量小于0: {train_size}/{val_size}/{test_size}")
            continue

        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        X_train_site = X_site[train_idx]
        Y_train_site = Y_site[train_idx]
        X_val_site = X_site[val_idx]
        Y_val_site = Y_site[val_idx]
        X_test_site = X_site[test_idx]
        Y_test_site = Y_site[test_idx]

        if print_on:
            print(f"站点 {site + 1} 训练集形状: {X_train_site.shape}, 验证集形状: {X_val_site.shape}, "
                f"测试集形状: {X_test_site.shape}")

        # 站点级别的输入标准化（！注意此处标准化方式，后续在Use中用）
        scaler_X_site = StandardScaler()

        # 训练集标准化
        X_train_reshaped = X_train_site.reshape(-1, 1 + num_meteo_features) # 标准化时形状需要啥二维
        X_train_site = scaler_X_site.fit_transform(X_train_reshaped).reshape(X_train_site.shape)

        # 应用标准化到验证和测试集
        X_val_site = scaler_X_site.transform(X_val_site.reshape(-1, 1 + num_meteo_features)).reshape(X_val_site.shape)
        X_test_site = scaler_X_site.transform(X_test_site.reshape(-1, 1 + num_meteo_features)).reshape(X_test_site.shape)

        # 存储当前站点数据
        site_datasets.append({
            'X_train': X_train_site, 'Y_train': Y_train_site,
            'X_val': X_val_site, 'Y_val': Y_val_site,
            'X_test': X_test_site, 'Y_test': Y_test_site
        })

        # 存储标准化器
        scaler_X_list.append(scaler_X_site)

        # 收集当前站点的Y数据（用于后续全局标准化）
        all_Y_list.append(Y_train_site.flatten())

        # 保存当前站点的标准化器
        os.makedirs(model_save_path, exist_ok=True)
        scaler_filename = os.path.join(model_save_path, f"{model_type}_{case_name}_scaler_X{site + 1}.pkl")
        dump(scaler_X_site, scaler_filename)
        if print_on:
            print(f"已保存站点 {site + 1} 的X标准化器到: {scaler_filename}")

        valid_sites += 1

    # 检查是否有有效站点
    if valid_sites == 0:
        raise ValueError("没有有效的站点数据可用")

    # 检查收集的Y数据是否为空
    if len(all_Y_list) == 0:
        raise ValueError("没有收集到任何Y数据用于全局标准化")

    # 合并所有Y数据
    all_Y_train = np.concatenate(all_Y_list).reshape(-1, 1)

    # 检查合并后的Y数据是否为空
    if all_Y_train.size == 0:
        raise ValueError("合并后的Y数据为空，无法进行全局标准化")

    # 应用Y标准化（根据参数选择方法）
    if y_scaling_method == 'global':
        # 全局标准化方法（原有方法）
        scaler_Y_global = StandardScaler()
        scaler_Y_global.fit(all_Y_train)

        # 保存全局Y标准化器
        scaler_y_filename = os.path.join(model_save_path, f"{model_type}_{case_name}_scaler_Y_global.pkl")
        dump(scaler_Y_global, scaler_y_filename)
        print(f"\n已保存全局Y标准化器到: {scaler_y_filename}")
        print(f"  scaler_Y_global，样本数: {scaler_Y_global.n_features_in_}, "
              f"均值={scaler_Y_global.mean_[0]:.2f}, 标准差={scaler_Y_global.scale_[0]:.2f}")

        # 应用全局Y标准化到每个站点的数据集
        for site_data in site_datasets:
            for key in ['Y_train', 'Y_val', 'Y_test']:
                Y_data = site_data[key]
                Y_reshaped = Y_data.reshape(-1, 1)
                site_data[key] = scaler_Y_global.transform(Y_reshaped).reshape(Y_data.shape)

    elif y_scaling_method == 'forecast_based':
        # 使用预报潮位的标准化参数
        print("\n使用原预测数据的标准化参数进行Y标准化")

        for site_idx, (site_data, scaler_X_site) in enumerate(zip(site_datasets, scaler_X_list)):
            # 获取预报潮位的标准化参数
            mean_forecast = scaler_X_site.mean_[0]  # 第一个特征的均值（预报潮位）
            std_forecast = scaler_X_site.scale_[0]  # 第一个特征的标准差

            if print_on:
                print(f"站点 {site_idx + 1} 预报潮位标准化参数: 均值={mean_forecast:.4f}, 标准差={std_forecast:.4f}")

            # 标准化Y数据
            for key in ['Y_train', 'Y_val', 'Y_test']:
                Y_data = site_data[key]
                # 标准化公式: (x - mean) / std
                site_data[key] = (Y_data - mean_forecast) / std_forecast

    else:
        raise ValueError(f"未知的Y标准化方法: {y_scaling_method}")

    # 合并所有站点的数据集
    def combine_site_data(key):
        return np.vstack([site[key] for site in site_datasets])

    # 创建合并后的数据集
    X_train_combined = combine_site_data('X_train')
    Y_train_combined = combine_site_data('Y_train')
    X_val_combined = combine_site_data('X_val')
    Y_val_combined = combine_site_data('Y_val')
    X_test_combined = combine_site_data('X_test')
    Y_test_combined = combine_site_data('Y_test')

    print(f"\n{'=' * 40} 最终数据集 {'=' * 40}")
    print(
        f"合并后输入数据形状 - 训练集: {X_train_combined.shape}, 验证集: {X_val_combined.shape}, 测试集: {X_test_combined.shape}")
    print(
        f"合并后输出数据形状 - 训练集: {Y_train_combined.shape}, 验证集: {Y_val_combined.shape}, 测试集: {Y_test_combined.shape}")

    # 为每个站点数据集添加原始站点索引信息
    for i, site_data in enumerate(site_datasets):
        site_data['site_index'] = i

    # 最终打乱合并后的数据集
    def shuffle_data(X, Y):
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        return X[idx], Y[idx]

    X_train_combined, Y_train_combined = shuffle_data(X_train_combined, Y_train_combined)
    X_val_combined, Y_val_combined = shuffle_data(X_val_combined, Y_val_combined)
    X_test_combined, Y_test_combined = shuffle_data(X_test_combined, Y_test_combined)

    return {
        # 合并后的数据集（用于训练）
        'X_train': X_train_combined, 'Y_train': Y_train_combined,
        'X_val': X_val_combined, 'Y_val': Y_val_combined,
        'X_test': X_test_combined, 'Y_test': Y_test_combined,

        # 各站点单独的数据集（用于SHAP分析）
        'site_datasets': site_datasets,

        # 标准化信息
        'scaler_X_list': scaler_X_list,
        'y_scaling_method': y_scaling_method,

        # 数据集信息
        'forward_length': forward_length,
        'backward_length': backward_length,
        'num_features': 1 + num_meteo_features,

        # 站点信息（方便识别）
        'valid_sites': valid_sites
    }