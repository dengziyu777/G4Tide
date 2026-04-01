import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm



def Fv1_evaluate_model_on_test_set_LSTM(model, test_loader, criterion, device, forecast_horizon):
    """
    在测试集上全面评估模型性能

    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备
        forecast_horizon: 未来时间步长

    返回:
        test_results: 测试结果字典
    """
    model.eval()
    all_predictions = []
    all_targets = []
    test_loss = 0.0
    test_batches = 0

    print("开始在测试集上评估模型...")

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="测试集评估"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)

            # 计算损失
            if isinstance(criterion, torch.nn.Module):
                loss = criterion(outputs, y_batch)
            else:
                loss = criterion(y_batch, outputs)

            test_loss += loss.item()
            test_batches += 1

            # 收集预测值和真实值
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    # 合并所有批次的结果
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # 计算平均测试损失
    avg_test_loss = test_loss / test_batches

    # 计算各种评估指标
    mae = mean_absolute_error(all_targets, all_predictions)
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)

    # 计算R²分数
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    # 计算每个预测步长的指标（如果是多步预测）
    step_metrics = {}
    if forecast_horizon > 1:
        step_mae = []
        step_rmse = []
        step_r2 = []

        for step in range(forecast_horizon):
            step_pred = all_predictions[:, step]
            step_true = all_targets[:, step]

            step_mae.append(mean_absolute_error(step_true, step_pred))
            step_rmse.append(np.sqrt(mean_squared_error(step_true, step_pred)))

            ss_res_step = np.sum((step_true - step_pred) ** 2)
            ss_tot_step = np.sum((step_true - np.mean(step_true)) ** 2)
            step_r2.append(1 - (ss_res_step / (ss_tot_step + 1e-8)))

        step_metrics = {
            'step_mae': step_mae,
            'step_rmse': step_rmse,
            'step_r2': step_r2
        }

    test_results = {
        'test_loss': avg_test_loss,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'predictions': all_predictions,
        'targets': all_targets,
        'step_metrics': step_metrics if forecast_horizon > 1 else None
    }

    print("测试集评估完成!")
    return test_results