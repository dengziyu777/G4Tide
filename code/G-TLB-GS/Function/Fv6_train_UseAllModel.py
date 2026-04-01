import sys
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt  # 添加绘图库
from sklearn.metrics import r2_score
from matplotlib.ticker import ScalarFormatter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from source.Fv6_safe_torch_load import Fv6_safe_torch_load


# 定义动态MAE损失函数，防止模型输出简单平均值
class DynamicMAELoss(nn.Module):
    def __init__(self, base_loss=nn.L1Loss(), var_weight=0.2, min_var=0.05):
        """
        动态MAE损失函数，通过方差惩罚防止模型仅预测平均值

        参数:
            base_loss: 基础损失函数 (默认为MAE)
            var_weight: 方差惩罚权重
            min_var: 最小期望方差阈值
        """
        super().__init__()
        self.base_loss = base_loss
        self.var_weight = var_weight
        self.min_var = min_var
        self.adaptive_weight = True

    def forward(self, outputs, targets):
        # 计算基础损失（MAE）
        base_loss = self.base_loss(outputs, targets)

        # 计算批次输出的方差
        batch_var = torch.var(outputs)

        # 动态方差惩罚机制
        if batch_var < self.min_var:
            # 当方差低于阈值时增加惩罚权重
            current_weight = self.var_weight * (1 + (self.min_var - batch_var) / self.min_var)
            var_penalty = current_weight * (self.min_var - batch_var)
        else:
            # 正常方差下不使用惩罚
            var_penalty = 0.0

        return base_loss + var_penalty


def Fv6_train_UseAllModel(model, model_type, train_loader, val_loader, model_save_path, case_name,
                          PTR, epochs, patience, min_delta):
    """
    训练适用于所有站点的单一模型（v6版本）
    250716：增加训练时评价指标为MAE、RMSE及R2；使用R²作为早停和学习率调整的主要指标
    250717：解决模型容易预测平均值的问题
    参数:
        model: 待训练的模型实例（可以是任何nn.Module子类）
        model_type: 模型类型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        model_save_path: 模型保存目录
        case_name: 案例名称(用于文件名)
        PTR: 每PTR轮次打印训练记录
        epochs: 最大训练轮数
        patience: 允许验证损失不改进的轮数
        min_delta: 视为改进的最小变化量
    """
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Fv6_train_UseAllModel-使用 {device} 训练所有站点")

    # 确保保存目录存在
    os.makedirs(model_save_path, exist_ok=True)

    # 构造模型保存路径
    model_filename = f"{model_type}_{case_name}.pth"
    model_fullpath = os.path.join(model_save_path, model_filename)

    # 保存模型配置
    config_filename = f"{model_type}_{case_name}_config.pkl"
    config_fullpath = os.path.join(model_save_path, config_filename)

    # 根据模型类型提取配置参数
    if model_type == 'TCN':
        model_config = {
            'input_size': model.input_size,
            'output_size': model.output_size,
            'num_channels': model.num_channels,
            'kernel_size': model.kernel_size,
            'dropout': model.dropout
        }
    elif model_type == 'LSTM':
        model_config = {
            'input_size': model.input_size,
            'hidden_sizes': model.hidden_sizes,
            'output_size': model.output_size,
            'bidirectional': model.bidirectional,
            'dropout': model.dropout
        }
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 保存模型配置
    dump(model_config, config_fullpath)
    print(f"已保存模型配置到: {config_fullpath}")

    # 训练配置
    criterion = DynamicMAELoss(
        base_loss=nn.L1Loss(),
        var_weight=0.2,
        min_var=0.05
    ).to(device)

    # 使用AdamW优化器（带权重衰减）
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 使用带重启的余弦退火学习率调度器；模拟了余弦函数的形状来调整学习率，帮助模型在训练过程中更好地收敛
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,      # 第一次重启周期
        T_mult=2,    # 周期倍增因子
        eta_min=1e-6 # 最小学习率
    )

    best_val_r2 = -np.inf  # R²初始值设为负无穷
    best_epoch = 0
    epochs_no_improve = 0
    early_stop = False

    # 训练历史记录
    train_losses = []
    val_maes = []  # 验证集MAE
    val_rmses = [] # 验证集RMSE
    val_r2s = []   # 验证集R²
    learning_rates = []  # 记录学习率变化
    grad_norms = []      # 记录梯度范数
    pred_vars = []       # 记录预测方差

    for epoch in range(epochs):
        if early_stop:
            print(f"在轮次 {epoch} 触发早停!")
            break

        # 记录当前学习率（在更新前记录）
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # ===== 训练阶段 =====
        model.train()
        train_loss = 0
        epoch_pred_vars = []  # 记录每个batch预测的方差
        batch_grad_norms = []  # 记录每个batch的梯度范数

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(x)

            # 计算并记录预测方差
            batch_var = torch.var(outputs).item()
            epoch_pred_vars.append(batch_var)

            # 计算损失
            loss = criterion(outputs, y)

            # 反向传播
            loss.backward()

            # 梯度裁剪和监测
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=0.5, norm_type=2
            ).item()
            batch_grad_norms.append(grad_norm)

            # 梯度消失检测
            if grad_norm < 1e-7:
                print(f"警告: 梯度消失! 轮次 {epoch}, 批次 {batch_idx}")
                optimizer.zero_grad()
                continue

            # 更新权重
            optimizer.step()

            # 更新学习率（每个batch后更新）
            scheduler.step(epoch + batch_idx / len(train_loader))

            # 累加损失
            train_loss += loss.item()

        # 记录epoch级统计
        avg_pred_var = np.mean(epoch_pred_vars)
        pred_vars.append(avg_pred_var)
        avg_grad_norm = np.mean(batch_grad_norms)
        grad_norms.append(avg_grad_norm)

        # ===== 验证阶段 =====
        model.eval()
        total_mae = 0.0
        total_mse = 0.0  # 用于计算RMSE
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)

                # 计算MAE和MSE
                total_mae += torch.sum(torch.abs(outputs - y)).item()
                total_mse += torch.sum((outputs - y) ** 2).item()

                # 收集预测和目标值用于R²计算
                all_targets.append(y.cpu().numpy())
                all_preds.append(outputs.cpu().numpy())

        # 计算MAE和RMSE
        num_val_samples = len(val_loader.dataset)
        val_mae = total_mae / num_val_samples
        val_rmse = np.sqrt(total_mse / num_val_samples)

        # 计算R²（带有鲁棒性处理）
        all_targets = np.concatenate(all_targets, axis=0).flatten()
        all_preds = np.concatenate(all_preds, axis=0).flatten()

        val_r2 = r2_score(all_targets, all_preds)

        # 记录验证指标
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        val_maes.append(val_mae)
        val_rmses.append(val_rmse)
        val_r2s.append(val_r2)

        # 每PTR轮打印进度
        if epoch % PTR == 0 or epoch == 0:
            print(
                f"轮次 {epoch:3d}/{epochs} | 训练损失: {avg_train_loss:.6f} | "
                f"验证MAE: {val_mae:.6f} | 验证RMSE: {val_rmse:.6f} | "
                f"验证R²: {val_r2:.6f} | 学习率: {current_lr:.6f} | "
                f"预测方差: {avg_pred_var:.4f} | 梯度范数: {avg_grad_norm:.2f}")

        # 使用R²作为早停指标
        if val_r2 > best_val_r2 + min_delta:
            best_val_r2 = val_r2
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_fullpath)
            print(f"轮次 {epoch:3d} | 保存新最佳模型 (验证R²: {val_r2:.6f}, "
                  f"验证MAE: {val_mae:.6f}, 预测方差: {avg_pred_var:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True
                # 加载最佳模型
                model.load_state_dict(Fv6_safe_torch_load(model_fullpath, map_location=device))
                print(f"早停: 恢复轮次 {best_epoch} 的最佳模型 (验证R²: {best_val_r2:.6f})")

    # 最终保存模型
    torch.save(model.state_dict(), model_fullpath)
    print(f"训练完成，最终模型已保存到: {model_fullpath}")

    # === 绘制综合训练曲线图 ===
    plt.figure(figsize=(18, 12))

    # 1. 训练和验证MAE曲线
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Training', color='#7200da', alpha=0.8)
    plt.plot(val_maes, label='Validation', color='#f9320c', alpha=0.8)
    plt.title(f'MAE', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE', fontsize=12)

    # 标记最佳验证损失点
    if best_epoch < len(val_maes):
        plt.scatter([best_epoch], [val_maes[best_epoch]], color='#00b9f1', zorder=5,
                    s=100, edgecolors='black', label=f'Best Model (Epoch {best_epoch})')
    plt.axvline(x=best_epoch, color='#00b9f1', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')  # 修改为左下角
    plt.grid(True, linestyle='--', alpha=0.3)

    # 3. 验证RMSE曲线
    plt.subplot(2, 3, 2)
    plt.plot(val_rmses, label='RMSE', color='#f9320c', alpha=0.8)
    plt.title(f'RMSE', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)

    # 标记最佳RMSE点
    plt.scatter([best_epoch], [val_rmses[best_epoch]], color='#00b9f1', zorder=5,
                s=100, label=f'Best Model (Epoch {best_epoch})')
    plt.axvline(x=best_epoch, color='#00b9f1', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')  # 修改为左下角
    plt.grid(True, linestyle='--', alpha=0.3)

    # 2. 验证R²曲线
    plt.subplot(2, 3, 3)
    plt.plot(val_r2s, label='R²', color='#f9320c', alpha=0.8)
    plt.title(f'R²', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('R²', fontsize=12)

    # 标记最佳R²点
    plt.scatter([best_epoch], [val_r2s[best_epoch]], color='#00b9f1', zorder=5,
                s=100, label=f'Best R² (Epoch {best_epoch})')
    plt.axvline(x=best_epoch, color='#00b9f1', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')  # 修改为左下角
    plt.grid(True, linestyle='--', alpha=0.3)

    # 4. 预测方差曲线
    plt.subplot(2, 3, 4)
    plt.plot(pred_vars, label='Prediction Variance', color='#f9320c', alpha=0.8)
    plt.title(f'Model Prediction Variance', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Variance', fontsize=12)
    plt.scatter([best_epoch], [pred_vars[best_epoch]], color='#00b9f1', zorder=5,
                s=100, label=f'Best Model Variance')
    plt.axvline(x=best_epoch, color='#00b9f1', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')  # 修改为左下角
    plt.grid(True, linestyle='--', alpha=0.3)

    # 5. 梯度范数曲线
    plt.subplot(2, 3, 5)
    plt.plot(grad_norms, label='Gradient Norm', color='#f9320c', alpha=0.8)
    plt.title(f'Gradient Norm', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Gradient Norm', fontsize=12)
    plt.yscale('log')
    plt.scatter([best_epoch], [grad_norms[best_epoch]], color='#00b9f1', zorder=5,
                s=100, label=f'Best Model Gradient')
    plt.axvline(x=best_epoch, color='#00b9f1', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')  # 修改为左下角
    plt.grid(True, linestyle='--', alpha=0.3)

    # 6. 学习率变化曲线
    plt.subplot(2, 3, 6)
    plt.plot(learning_rates, label='Learning Rate', color='#f9320c', alpha=0.8)
    plt.title('Learning Rate Schedule', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.yscale('log')
    plt.scatter([best_epoch], [learning_rates[best_epoch]], color='#00b9f1', zorder=5,
                s=100, label=f'Best Model LR')
    plt.axvline(x=best_epoch, color='#00b9f1', linestyle='--', alpha=0.5)
    plt.legend(loc='lower left')  # 修改为左下角
    plt.grid(True, linestyle='--', alpha=0.3, which='both')

    plt.suptitle(f'{model_type} Model Training Metrics - {case_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])


    # 保存损失曲线图
    plot_filename = f"{model_type}_{case_name}_metrics_curve.png"
    plot_fullpath = os.path.join(model_save_path, plot_filename)
    plt.savefig(plot_fullpath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"训练完成，已保存综合指标曲线图到: {plot_fullpath}")

    # 返回训练历史
    return {
        'train_losses': train_losses,
        'val_maes': val_maes,
        'val_rmses': val_rmses,
        'val_r2s': val_r2s,
        'learning_rates': learning_rates,
        'grad_norms': grad_norms,
        'pred_vars': pred_vars,
        'best_epoch': best_epoch,
        'best_val_r2': best_val_r2,
        'metrics_plot_path': plot_fullpath
    }