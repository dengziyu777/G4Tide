import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def Fv1_evaluate_final_metrics_LSTM(model, train_loader, val_loader, criterion, device):
    """
    计算模型在训练集和验证集上的最终指标
    """

    def evaluate_loader(data_loader, loader_name):
        model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0
        batches = 0

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)

                if isinstance(criterion, torch.nn.Module):
                    loss = criterion(outputs, y_batch)
                else:
                    loss = criterion(y_batch, outputs)

                total_loss += loss.item()
                batches += 1
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        avg_loss = total_loss / batches
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        return {
            'loss': avg_loss,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

    print("计算训练集和验证集最终指标...")
    train_metrics = evaluate_loader(train_loader, "训练集")
    val_metrics = evaluate_loader(val_loader, "验证集")

    return train_metrics, val_metrics