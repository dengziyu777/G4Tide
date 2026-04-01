import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def Fv1_evaluate_model_LSTM(model, test_loader, criterion, device, site_info):
    """
    评估模型性能

    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备
        site_info: 站点信息（用于反标准化）

    返回:
        test_loss: 测试损失
        mae: 平均绝对误差
        rmse: 均方根误差
        all_predictions: 所有预测值
        all_actuals: 所有真实值
    """
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()

            # 收集预测值和真实值
            all_predictions.extend(outputs.cpu().numpy())
            all_actuals.extend(y_batch.cpu().numpy())

    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    # 计算MAE和RMSE
    mae = mean_absolute_error(all_actuals, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))

    # 计算平均测试损失
    avg_test_loss = test_loss / len(test_loader)

    return avg_test_loss, mae, rmse, all_predictions, all_actuals