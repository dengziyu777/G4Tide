import os
import torch
import numpy as np
from joblib import load
import sys
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from source.Fv6_safe_torch_load import Fv6_safe_torch_load
from source.Fv6_TCNModel import TCNModel
from source.Fv6_LSTMModel import LSTMModel


def Fv6_predict_NewF_ScaleXIndep(sequences, model_folder_path, model_type, case_name, device, batch_size, site_idx, y_scaling_method, print_on = True):
    """
    基于训练好的深度学习模型生成新的预测潮位数据

    参数:
        sequences: 当前站点的输入序列数据 [样本数, 历史长度, 特征数]
        model_folder_path: 模型文件存储路径
        model_type: 模型类型 ('TCN' 或 'LSTM')
        case_name: 案例名称，用于构建文件名
        device: 计算设备 ('cuda' 或 'cpu')
        batch_size: 批处理尺寸
        site_idx: 站点索引，用于文件名
        y_scaling_method：Y标准化方法，global为基于全部观测数据统一标准化、forecast_based为基于各站点预测数据分别标准化
        print_on：执行函数时，是否打印信息

    返回:
        predictions: 当前站点的预测结果数组 [样本数]
    """
    try:
        if print_on:
            print(f"  站点 {site_idx} 输入序列形状: {sequences.shape} (样本数×历史长度×特征数)")

        # 1. 准备模型和scaler文件路径
        model_file = f"{model_type}_{case_name}.pth"
        model_path = os.path.join(model_folder_path, model_file)

        config_file = f"{model_type}_{case_name}_config.pkl"
        config_path = os.path.join(model_folder_path, config_file)

        # 2. 加载模型配置
        model_config = load(config_path)
        if print_on:
            print(f"  训练时使用的Y标准化方法: {y_scaling_method}")

        # 3. 根据Y标准化方法选择反标准化方式
        if y_scaling_method == 'global':
            # 加载全局Y标准化器
            scalerY_file = f"{model_type}_{case_name}_scaler_Y_global.pkl"
            scalerY_path = os.path.join(model_folder_path, scalerY_file)

            if not os.path.exists(scalerY_path):
                raise FileNotFoundError(f"全局Y标准化器文件不存在: {scalerY_path}")

            scalerY = load(scalerY_path)
            if print_on:
                print(f"  已加载全局Y标准化器，特征数: {scalerY.n_features_in_},均值={scalerY.mean_[0]:.2f},标准差={scalerY.scale_[0]:.2f}")

        elif y_scaling_method == 'forecast_based':
            # 直接从输入数据计算预报潮位的统计量
            # 假设预报潮位是第一特征
            forecast_values = sequences[:, 0, 0].flatten()  # 当前站点 提取样本数、第一个时间步、潮位值
            mean_forecast = np.mean(forecast_values)
            std_forecast = np.std(forecast_values)
            if print_on:
                print(f"  计算输入层预报数据统计量: 均值={mean_forecast:.4f}, 标准差={std_forecast:.4f}")

        else:
            raise ValueError(f"不支持的Y标准化方法: {y_scaling_method}")

        # 4. 加载模型配置并重建模型
        if model_type == 'TCN':
            model = TCNModel(
                input_size=model_config['input_size'],
                output_size=model_config['output_size'],
                num_channels=model_config['num_channels'],
                kernel_size=model_config['kernel_size'],
                dropout=model_config['dropout']
            )
        elif model_type == 'LSTM':
            model = LSTMModel(
                input_size=model_config['input_size'],
                hidden_sizes=model_config['hidden_sizes'],
                output_size=model_config['output_size'],
                bidirectional=model_config['bidirectional'],
                dropout=model_config['dropout']
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 5. 加载模型权重
        model.load_state_dict(Fv6_safe_torch_load(model_path, device))
        model.to(device)
        model.eval()
        if print_on:
            print(f"  已加载模型权重: {model_path}")

        # 6. 标准化输入数据
        sequences_shape = sequences.shape

        # 只提取每个样本的第一个时间步（index=0）
        first_timestep_data = sequences[:, 0, :]  # 提取所有样本的第一个时间步
        input_means = np.mean(first_timestep_data, axis=0)  # axis=0对每一列（每个特征）的所有行（所有样本）计算平均值
        input_stds = np.std(first_timestep_data, axis=0)

        # 安全标准化处理
        epsilon = 1e-8  # 防止除零的极小值
        input_stds_safe = np.where(input_stds < epsilon, epsilon, input_stds)  # 替换过小的标准差

        # 对整个序列进行标准化（包括所有时间步）
        sequences_flat = sequences.reshape(-1, sequences_shape[2])
        sequences_scaled = (sequences_flat - input_means) / input_stds_safe
        sequences_scaled = sequences_scaled.reshape(sequences_shape)

        if print_on:
            print(f"  已计算输入特征统计量:")
            for i, (mean, std, stds_safe) in enumerate(zip(input_means, input_stds, input_stds_safe)):
                print(f"    站点{site_idx} - 特征 {i}: 均值={mean:.4e}, 标准差={std:.4e}， 安全标准差={stds_safe:.4e}")

        # 7. 准备输入数据
        X_tensor = torch.tensor(sequences_scaled, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # 7. 进行预测
        y_outputs = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device)
                outputs = model(x)
                y_outputs.append(outputs.cpu().numpy())

        y_outputs = np.concatenate(y_outputs)

        # 9. 反标准化预测结果
        if y_scaling_method == 'global':
            final_preds = scalerY.inverse_transform(y_outputs).flatten()
        else:  # forecast_based
            # 反标准化公式: y = y_std * std_forecast + mean_forecast
            final_preds = y_outputs * std_forecast + mean_forecast
            final_preds = final_preds.flatten()

        if print_on:
            print(f"  成功生成 {len(final_preds)} 个预测值")
        return final_preds

    except Exception as e:
        print(f"站点 {site_idx} 预测失败: {str(e)}")
        sys.exit(1)