import pandas as pd


def Fv1_save_results_to_txt_LSTM(training_history, test_results, train_metrics, val_metrics,model_config, file_path):
    """
    将训练、验证和测试结果保存到txt文件

    参数:
        training_history: 训练历史记录
        test_results: 测试集结果
        train_metrics: 训练集指标
        val_metrics: 验证集指标
        model_config: 模型配置参数
        file_path: 保存文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("           标准LSTM模型评估报告\n")
        f.write("=" * 60 + "\n\n")

        # 1. 模型配置信息
        f.write("1. 模型配置参数\n")
        f.write("-" * 40 + "\n")
        for key, value in model_config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        # 2. 训练过程摘要
        f.write("2. 训练过程摘要\n")
        f.write("-" * 40 + "\n")
        f.write(f"最佳验证损失: {training_history.get('best_val_loss', 'N/A'):.6f}\n")
        f.write(f"最佳模型轮次: {training_history.get('best_epoch', 'N/A') + 1}\n")
        f.write(f"总训练轮次: {len(training_history.get('train_losses', []))}\n")
        f.write(f"最终学习率: {training_history.get('learning_rates', [])[-1] if training_history.get('learning_rates') else 'N/A':.2e}\n")
        f.write("\n")

        # 3. 数据集性能指标
        f.write("3. 各数据集性能指标\n")
        f.write("-" * 40 + "\n")

        # 训练集指标
        f.write("训练集:\n")
        f.write(f"  损失 (MSE): {train_metrics['loss']:.6f}\n")
        f.write(f"  MAE: {train_metrics['mae']:.6f}\n")
        f.write(f"  RMSE: {train_metrics['rmse']:.6f}\n")
        f.write(f"  R²: {train_metrics['r2']:.6f}\n\n")

        # 验证集指标
        f.write("验证集:\n")
        f.write(f"  损失 (MSE): {val_metrics['loss']:.6f}\n")
        f.write(f"  MAE: {val_metrics['mae']:.6f}\n")
        f.write(f"  RMSE: {val_metrics['rmse']:.6f}\n")
        f.write(f"  R²: {val_metrics['r2']:.6f}\n\n")

        # 测试集指标
        f.write("测试集:\n")
        f.write(f"  损失 (MSE): {test_results['test_loss']:.6f}\n")
        f.write(f"  MAE: {test_results['mae']:.6f}\n")
        f.write(f"  RMSE: {test_results['rmse']:.6f}\n")
        f.write(f"  R²: {test_results['r2']:.6f}\n\n")

        # 4. 多步预测的逐步指标（如果适用）
        if test_results['step_metrics'] is not None:
            f.write("4. 多步预测逐步指标\n")
            f.write("-" * 40 + "\n")
            step_metrics = test_results['step_metrics']
            for step in range(len(step_metrics['step_mae'])):
                f.write(f"步长 {step + 1}:\n")
                f.write(f"  MAE: {step_metrics['step_mae'][step]:.6f}\n")
                f.write(f"  RMSE: {step_metrics['step_rmse'][step]:.6f}\n")
                f.write(f"  R²: {step_metrics['step_r2'][step]:.6f}\n")
            f.write("\n")

        # 5. 训练历史（最后几轮）
        f.write("5. 训练历史（最后5轮）\n")
        f.write("-" * 40 + "\n")
        train_losses = training_history.get('train_losses', [])
        val_losses = training_history.get('val_losses', [])

        if len(train_losses) > 0:
            f.write("轮次 | 训练损失 | 验证损失\n")
            f.write("-" * 30 + "\n")
            start_idx = max(0, len(train_losses) - 5)
            for i in range(start_idx, len(train_losses)):
                f.write(f"{i + 1:4d} | {train_losses[i]:.6f} | {val_losses[i]:.6f}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("评估完成时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("=" * 60 + "\n")

    print(f"结果已保存到: {file_path}")