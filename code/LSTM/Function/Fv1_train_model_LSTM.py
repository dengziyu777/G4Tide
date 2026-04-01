import torch
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import CosineAnnealingLR


def Fv1_train_model_LSTM(model, train_loader, val_loader, criterion, optimizer, max_epochs, patience, device,
                         model_save_dir, model_name):
    """
    训练LSTM模型

    参数:
        model: LSTM模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion：损失函数
        optimizer: 优化器
        max_epochs: 最大训练轮数
        patience: 早停耐心值
        device: 训练设备
        model_save_dir: 模型保存目录
        model_name: 模型保存名称

    返回:
        model: 训练好的模型
        train_losses: 训练损失记录
        val_losses: 验证损失记录
        best_epoch: 最佳模型对应的轮次
    """
    model.to(device)

    # 创建模型保存目录
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"{model_name}.pth")

    # 记录训练过程
    train_losses = []
    val_losses = []
    learning_rates = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    # 早停的最小改善阈值（R2）
    min_delta = 0.001

    # 进度条
    epoch_pbar = tqdm(range(max_epochs), desc="训练进度")

    for epoch in epoch_pbar:
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(X_batch)

            loss = criterion(y_batch, outputs)

            # 反向传播
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        # 获取当前固定学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # 计算平均训练损失
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)

                if isinstance(criterion, torch.nn.Module):
                    loss = criterion(outputs, y_batch)
                else:
                    loss = criterion(y_batch, outputs)

                val_loss += loss.item()
                val_batches += 1

        # 计算平均验证损失
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)

        # 计算R²分数（此处仅用于显示，不用于优化）
        with torch.no_grad():
            # 使用验证集计算R²
            all_preds = []
            all_targets = []
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                all_preds.append(outputs.cpu())
                all_targets.append(y_batch.cpu())

            all_preds = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            val_r2 = 1 - torch.sum((all_targets - all_preds) ** 2) / torch.sum(
                (all_targets - torch.mean(all_targets)) ** 2)   # 组内平均/整体平均 均近似为平均海平面
            val_r2 = val_r2.item()

        # 更新进度条描述
        epoch_pbar.set_postfix({
            '训练损失': f'{avg_train_loss:.4f}',    # 损失基于1-R2进行评价
            '验证损失': f'{avg_val_loss:.4f}',
            '验证R²': f'{val_r2:.4f}',
            '学习率': f'{current_lr:.2e}',
            '早停计数': f'{patience_counter}/{patience}'
        })

        # 早停检查（包含最小改善阈值）
        if avg_val_loss < best_val_loss - min_delta:  # 只有当改善超过min_delta时才更新
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            # 保存最佳模型状态
            best_model_state = model.state_dict().copy()
            # 保存最佳模型到指定目录
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'val_r2': val_r2,
                'optimizer_state_dict': optimizer.state_dict()
            }, model_save_path)
            print(f"\n✓ 最佳模型已保存到: {model_save_path}")
        else:
            patience_counter += 1

        # 检查是否早停
        if patience_counter >= patience:
            print(f"\n早停于第 {epoch + 1} 轮，最佳模型在第 {best_epoch + 1} 轮")
            print(f"验证损失改善未超过 {min_delta} 持续 {patience} 轮")
            break

        # 每10轮打印一次详细信息
        if (epoch + 1) % 10 == 0:
            print(f'轮次 {epoch + 1}/{max_epochs}, '
                  f'训练损失: {avg_train_loss:.4f}, '
                  f'验证损失: {avg_val_loss:.4f}, '
                  f'验证R²: {val_r2:.4f}, '
                  f'学习率: {current_lr:.2e}')

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"已加载第 {best_epoch + 1} 轮的最佳模型")

    print(f"训练完成! 最佳验证损失: {best_val_loss:.4f} (第{best_epoch + 1}轮)")

    # 返回训练历史
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'model_save_path': model_save_path
    }

    return model, training_history