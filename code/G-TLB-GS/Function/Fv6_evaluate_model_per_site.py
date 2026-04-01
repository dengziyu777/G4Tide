import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from source.Fv6_safe_torch_load import Fv6_safe_torch_load
from source.Fv6_TCNModel import TCNModel  # 根据实际位置导入
from source.Fv6_LSTMModel import LSTMModel  # 根据实际位置导入


def Fv6_evaluate_model_per_site(combined_data, model_save_path, output_file, model_type, case_name, time_interval):
    """
    分站点评估模型在测试集上的表现

    参数:
        combined_data: 合并后的数据对象（包含站点数据集和元数据）
        model_save_path: 模型保存目录
        output_file: 评估结果文件路径
        model_type: 模型类型
        case_name: 案例名称
        time_interval: 时间步长（用于记录）

    返回:
        site_metrics: 站点评估结果列表
    """
    # 初始化评估结果文件
    scaling_method = combined_data.get('y_scaling_method', 'global')    # 尝试从 combined_data 中获取 'y_scaling_method' 的值；如果该键存在，则将其值赋给 scaling_method 变量；如果该键不存在，则将 'global' 赋给 scaling_method 变量
    Fv6_write_evaluation_metrics_header(
        output_file,
        note=f"\n9.模型在测试集上的评估结果（取公共时段、观测数据时间步长{time_interval}s) - 标准化方法: {scaling_method}"
    )

    # 设置评估设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型文件路径
    model_filename = f"{model_type}_{case_name}.pth"
    model_fullpath = os.path.join(model_save_path, model_filename)
    model_config_name = f"{model_type}_{case_name}_config.pkl"
    model_config_path = os.path.join(model_save_path, model_config_name)

    # 初始化站点指标列表
    site_metrics = []
    valid_metrics = []
    total_samples = 0  # 记录总样本数

    # 遍历每个站点的数据集
    for site_idx, site_data in enumerate(combined_data['site_datasets']):
        try:
            print(f"\n{'=' * 40} 站点 {site_idx + 1} 评估开始 {'=' * 40}")

            # 获取当前站点的测试数据
            X_test_site = site_data['X_test']
            Y_test_site = site_data['Y_test']
            num_samples = len(Y_test_site)  # 当前站点样本数量
            total_samples += num_samples
            print(f"测试样本数量: {num_samples}")

            # 创建当前站点的测试数据加载器
            test_dataset = TensorDataset(
                torch.tensor(X_test_site, dtype=torch.float32),
                torch.tensor(Y_test_site, dtype=torch.float32)
            )
            test_loader = DataLoader(test_dataset, batch_size=len(X_test_site), shuffle=False)

            # 加载模型
            model = Fv6_load_model_from_config(
                model_type, model_fullpath, model_config_path, device
            )

            # 获取Y的反标准化器
            denormalizer = Fv6_get_denormalizer(
                model_type, case_name, scaling_method,
                site_idx, model_save_path
            )

            # 评估当前站点
            site_result = Fv6_evaluate_single_site(
                model, test_loader, denormalizer, device
            )

            # 保存评估结果
            site_metrics.append({
                'site_idx': site_idx,
                'mae': site_result['mae'],
                'rmse': site_result['rmse'],
                'r2': site_result['r2'],
                'y_true': site_result['y_true'],
                'y_pred': site_result['y_pred']
            })
            valid_metrics.append(site_result)

            print(f"站点 {site_idx + 1} 评估结果: "
                  f"MAE={site_result['mae']:.4f}, "
                  f"RMSE={site_result['rmse']:.4f}, "
                  f"R²={site_result['r2']:.4f}")

            # 保存单个站点评估结果到文件
            Fv6_write_site_evaluation_result(
                output_file, site_idx,num_samples,
                site_result['mae'], site_result['rmse'], site_result['r2']
            )

        except Exception as e:
            print(f"站点 {site_idx + 1} 评估失败: {str(e)}")
            site_metrics.append({
                'site_idx': site_idx,
                'error': str(e)
            })
            # 保存错误信息到文件
            Fv6_write_site_error_result(output_file, site_idx, str(e))

    # 计算并保存整体评估指标
    if valid_metrics:
        Fv6_calculate_and_save_overall_metrics(
            output_file, valid_metrics, site_metrics, total_samples
        )

    return site_metrics


def Fv6_load_model_from_config(model_type, model_path, config_path, device):
    """
    从配置文件加载模型

    参数:
        model_type: 模型类型
        model_path: 模型文件路径
        config_path: 配置文件路径
        device: 设备

    返回:
        model: 加载并准备好用于评估的模型
    """
    # 加载模型配置
    model_config = load(config_path)
    print(f"成功加载模型配置: {config_path}")

    # 根据模型类型重建模型
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
        raise ValueError(f"未知模型类型: {model_type}")

    # 加载模型权重
    model.load_state_dict(Fv6_safe_torch_load(model_path, device))
    model.to(device)
    model.eval()
    print(f"成功加载模型权重: {model_path}")

    return model


def Fv6_get_denormalizer(model_type, case_name, scaling_method, site_idx, model_save_path):
    """
    获取反标准化器

    参数:
        model_type: 模型类型
        case_name: 案例名称
        scaling_method: 标准化方法
        site_idx: 站点索引
        model_save_path: 模型保存目录

    返回:
        denormalize: 反标准化函数
    """
    if scaling_method == 'global':
        # 加载全局Y标准化器
        scaler_Y_name = f"{model_type}_{case_name}_scaler_Y_global.pkl"
        scaler_Y_path = os.path.join(model_save_path, scaler_Y_name)
        scaler_Y = load(scaler_Y_path)
        print(f"加载全局Y标准化器: {scaler_Y_path}")

        def denormalize(y):
            original_shape = y.shape
            flat_y = y.reshape(-1, 1)
            denorm_y = scaler_Y.inverse_transform(flat_y)
            return denorm_y.reshape(original_shape)

        return denormalize

    elif scaling_method == 'forecast_based':
        # 加载X标准化器
        scaler_X_name = f"{model_type}_{case_name}_scaler_X{site_idx + 1}.pkl"  # 站点编号从1开始
        scaler_X_path = os.path.join(model_save_path, scaler_X_name)
        scaler_X = load(scaler_X_path)
        print(f"加载站点 {site_idx + 1} 的X标准化器: {scaler_X_path}")

        # 获取预报数据的标准化参数
        mean_forecast = scaler_X.mean_[0]   # 选用预报数据中第一个特征的平均值
        std_forecast = scaler_X.scale_[0]   # 标准差
        print(f"  预报潮位标准化参数: 均值={mean_forecast:.4f}, 标准差={std_forecast:.4f}")

        def denormalize(y):
            return y * std_forecast + mean_forecast

        return denormalize

    else:
        raise ValueError(f"未知的标准化方法: {scaling_method}")


def Fv6_evaluate_single_site(model, test_loader, denormalizer, device):
    """
    评估单个站点

    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        denormalizer: 反标准化函数
        device: 设备

    返回:
        dict: 包含评估结果和反标准化后的数据
    """
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        total_samples = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            batch_samples = y.shape[0]
            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
            total_samples += batch_samples

    # 合并结果
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # 验证样本数量
    num_samples = y_true.shape[0]
    assert num_samples == total_samples, "样本数量不一致"

    # 反标准化后再评估
    y_true_denorm = denormalizer(y_true)
    y_pred_denorm = denormalizer(y_pred)

    # 计算评估指标
    mae = mean_absolute_error(y_true_denorm, y_pred_denorm)
    rmse = np.sqrt(mean_squared_error(y_true_denorm, y_pred_denorm))
    r2 = r2_score(y_true_denorm, y_pred_denorm)

    return {
        'num_samples': num_samples,  # 返回样本数量
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_true': y_true_denorm,
        'y_pred': y_pred_denorm
    }


# ================== 结果保存函数 ==================

def Fv6_write_evaluation_metrics_header(output_file, note):
    """
    写入评估结果文件头部信息
    """
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"{note}\n")
        f.write("Site\tMAE\tRMSE\tR²\tSampleCount\n")


def Fv6_write_site_evaluation_result(output_file, site_idx, num_samples, mae, rmse, r2):
    """
    写入单个站点的评估结果
    """
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"{site_idx + 1}\t{mae:.4f}\t{rmse:.4f}\t{r2:.4f}\t{num_samples}\n")


def Fv6_write_site_error_result(output_file, site_idx, error):
    """
    写入单个站点的错误信息
    """
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"{site_idx + 1}\t评估失败: {error}\n")


def Fv6_calculate_and_save_overall_metrics(output_file, valid_metrics, all_metrics, total_samples):
    """
    计算并保存整体评估指标
    """
    # 计算整体评估指标（按样本加权平均）
    if valid_metrics:
        # 计算加权平均指标
        total_valid_samples = sum(m['num_samples'] for m in valid_metrics)
        weighted_mae = sum(m['mae'] * m['num_samples'] for m in valid_metrics) / total_valid_samples
        weighted_rmse = sum(m['rmse'] * m['num_samples'] for m in valid_metrics) / total_valid_samples
        weighted_r2 = sum(m['r2'] * m['num_samples'] for m in valid_metrics) / total_valid_samples

        # 计算普通平均指标
        avg_mae = np.mean([m['mae'] for m in valid_metrics])
        avg_rmse = np.mean([m['rmse'] for m in valid_metrics])
        avg_r2 = np.mean([m['r2'] for m in valid_metrics])

        # 计算有效站点数量
        valid_count = len(valid_metrics)
        total_count = len(all_metrics)

        print(f"\n所有站点评估结果:")
        print(f"  - 样本总数: {total_samples}")
        print(f"  - 有效样本数: {total_valid_samples}")
        print(f"  - 加权平均: MAE={weighted_mae:.4f}, RMSE={weighted_rmse:.4f}, R²={weighted_r2:.4f}")
        print(f"  - 站点平均: MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, R²={avg_r2:.4f}")
        print(f"  - 有效站点数: {valid_count}/{total_count}")

    # 保存整体评估结果
    with open(output_file, 'a', encoding='utf-8') as f:
        if valid_count != total_count:
            f.write(f"注意: 部分站点评估失败 ({total_count - valid_count}个)\n")
        f.write(f"总样本数: {total_samples}\n")

        if valid_metrics:
            # 输出加权平均指标
            f.write(f"加权平均 (按样本加权)\t{weighted_mae:.4f}\t{weighted_rmse:.4f}\t{weighted_r2:.4f}\n")
            # 输出普通站点平均指标
            f.write(f"站点平均 (未加权)\t{avg_mae:.4f}\t{avg_rmse:.4f}\t{avg_r2:.4f}\n")