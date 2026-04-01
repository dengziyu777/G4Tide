import os
import traceback
import torch
import numpy as np
from joblib import load
import sys
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from source.Fv6_safe_torch_load import Fv6_safe_torch_load
from source.Fv6_TCNModel import TCNModel
from source.Fv6_LSTMModel import LSTMModel


def Fv6U2_predict_NewF_ScaleXIndep(sequences, model_folder_path, model_type, case_name, device, batch_size, site_idx,
                                 y_scaling_method):
    """
    基于训练好的深度学习模型生成新的预测潮位数据

    参数:
        sequences: 当前站点的输入序列数据 [样本数, 序列长度, 特征数]
        model_folder_path: 模型文件存储路径
        model_type: 模型类型 ('TCN' 或 'LSTM')
        case_name: 案例名称，用于构建文件名
        device: 计算设备 ('cuda' 或 'cpu')
        batch_size: 批处理尺寸
        site_idx: 站点索引，用于文件名
        y_scaling_method：Y标准化方法，global为基于全部观测数据统一标准化、forecast_based为基于各站点预测数据分别标准化

    返回:
        predictions: 当前站点的预测结果数组 [样本数, 未来时间步]
    """

    try:
        print(f"  站点 {site_idx} 输入序列形状: {sequences.shape} (样本数×历史长度×特征数)")

        # 1. 准备模型和scaler文件路径
        model_file = f"{model_type}_{case_name}.pth"
        model_path = os.path.join(model_folder_path, model_file)

        config_file = f"{model_type}_{case_name}_config.pkl"
        config_path = os.path.join(model_folder_path, config_file)

        # 2. 加载模型配置
        model_config = load(config_path)
        print(f"  训练时使用的Y标准化方法: {y_scaling_method}")

        # 获取未来步长（从配置中）
        future_steps = model_config['output_size']
        # print(f"  模型输出维度（未来步长）: {future_steps}")

        # 3. 根据Y标准化方法选择反标准化方式
        if y_scaling_method == 'global':
            # 加载全局Y标准化器
            scalerY_file = f"{model_type}_{case_name}_scaler_Y_global.pkl"
            scalerY_path = os.path.join(model_folder_path, scalerY_file)

            if not os.path.exists(scalerY_path):
                raise FileNotFoundError(f"全局Y标准化器文件不存在: {scalerY_path}")

            scalerY = load(scalerY_path)
            # 检查标准化器是否支持多步输出
            if scalerY.n_features_in_ != future_steps:
                raise ValueError(f"标准化器特征数({scalerY.n_features_in_})与模型输出维度({future_steps})不匹配")

            print(f"  已加载全局Y标准化器，支持{scalerY.n_features_in_}个输出特征")

        elif y_scaling_method == 'forecast_based':
            # 直接从输入数据计算预报潮位的统计量
            # 假设预报潮位是第一特征
            forecast_values = sequences[:, :, 0].flatten()  # 提取所有预报潮位值
            mean_forecast = np.mean(forecast_values)
            std_forecast = np.std(forecast_values)
            print(f"  计算输入层预报数据统计量: 均值={mean_forecast:.4f}, 标准差={std_forecast:.4f}")

        else:
            raise ValueError(f"不支持的Y标准化方法: {y_scaling_method}")

        # 4. 加载模型配置并重建模型
        if model_type == 'TCN':
            model = TCNModel(
                input_size=model_config['input_size'],
                output_size=future_steps,  # 确保输出维度匹配
                num_channels=model_config['num_channels'],
                kernel_size=model_config['kernel_size'],
                dropout=model_config['dropout']
            )
        elif model_type == 'LSTM':
            model = LSTMModel(
                input_size=model_config['input_size'],
                hidden_sizes=model_config['hidden_sizes'],
                output_size=future_steps,  # 确保输出维度匹配
                bidirectional=model_config['bidirectional'],
                dropout=model_config['dropout']
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # 5. 加载模型权重
        model.load_state_dict(Fv6_safe_torch_load(model_path, device))
        model.to(device)
        model.eval()
        print(f"  已加载模型权重: {model_path}")

        # 6. 标准化输入数据
        sequences_shape = sequences.shape
        sequences_flat = sequences.reshape(-1, sequences_shape[2])

        # 计算输入数据的均值和标准差
        input_means = np.mean(sequences_flat, axis=0)
        input_stds = np.std(sequences_flat, axis=0)

        # 标准化输入数据
        sequences_scaled = (sequences_flat - input_means) / input_stds
        sequences_scaled = sequences_scaled.reshape(sequences_shape)
        # print(f"  已标准化输入数据")

        # 7. 准备输入数据
        X_tensor = torch.tensor(sequences_scaled, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # 8. 进行预测
        y_outputs = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device)
                outputs = model(x)
                y_outputs.append(outputs.cpu().numpy())

        # 合并批次结果
        if len(y_outputs) > 0:
            y_outputs = np.vstack(y_outputs)
        else:
            y_outputs = np.array([])

        # 9. 反标准化预测结果
        if y_scaling_method == 'global':
            if y_outputs.size > 0:
                final_preds = scalerY.inverse_transform(y_outputs)
            else:
                final_preds = np.array([])
        else:  # forecast_based
            if y_outputs.size > 0:
                # 对每个时间步应用相同的反标准化
                final_preds = y_outputs * std_forecast + mean_forecast
            else:
                final_preds = np.array([])

        if y_outputs.size > 0:
            print(f"  预测结果形状: {final_preds.shape} (样本数×未来步长={final_preds.shape[0]}×{final_preds.shape[1]})")
        return final_preds

    except Exception as e:
        print(f"站点 {site_idx} 预测失败: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)