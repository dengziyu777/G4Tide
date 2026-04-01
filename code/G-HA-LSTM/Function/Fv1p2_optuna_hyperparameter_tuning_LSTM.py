import os
import torch
import optuna
from torch.optim import Adam

def objective(trial, forecast, train_loader, val_loader, test_loader,
              criterion, device, max_epochs, patience, model_save_dir,
              model_name_LSTM, lookback, train_ratio, val_ratio, batch_size,
              random_seed, ImproveLSTM, Fv1_train_model_LSTM,
              Fv1_evaluate_model_on_test_set_LSTM, Fv1_evaluate_final_metrics_LSTM,
              Fv1_save_results_to_txt_LSTM, num_layers_range, hidden_size_range,
              learning_rate_range, dropout_range):
    """
    Optuna 目标函数：定义要优化的目标（最小化验证损失，同时记录R²）
    """
    # 从 trial 中获取超参数建议
    num_layers = trial.suggest_categorical('num_layers', num_layers_range)
    hidden_size = trial.suggest_categorical('hidden_size', hidden_size_range)
    learning_rate = trial.suggest_categorical('learning_rate', learning_rate_range)
    dropout = trial.suggest_categorical('dropout', dropout_range)

    print(f"试验 #{trial.number}: 层数={num_layers}, 隐藏单元={hidden_size}, "
          f"学习率={learning_rate:.0e}, dropout={dropout:.2f}")

    try:
        # 调用训练函数
        best_val_loss, trained_model, training_history = train_with_hyperparams(
            num_layers, hidden_size, learning_rate, dropout, forecast,
            train_loader, val_loader, test_loader, criterion,
            device, max_epochs, patience, model_save_dir,
            model_name_LSTM, lookback, train_ratio, val_ratio,
            batch_size, random_seed, ImproveLSTM,
            Fv1_train_model_LSTM, Fv1_evaluate_model_on_test_set_LSTM,
            Fv1_evaluate_final_metrics_LSTM, Fv1_save_results_to_txt_LSTM
        )

        # 记录试验的额外信息
        trial.set_user_attr('training_history', training_history)   # 训练损失为1-R2
        trial.set_user_attr('model_path', os.path.join(
            model_save_dir, f"{num_layers}layers_{hidden_size}units_lr{learning_rate:.0e}_do{dropout:.2f}",
            f"{model_name_LSTM}_{num_layers}layers_{hidden_size}units_lr{learning_rate:.0e}_do{dropout:.2f}.pth"
        ))

        return best_val_loss    # ← 这是 Optuna 优化的目标

    except Exception as e:
        print(f"试验 #{trial.number} 失败: {e}")
        # 对于失败的试验，返回一个很大的损失值
        return float('inf')


def Fv1_hyperparameter_search_optuna_LSTM(num_layers_range, hidden_size_range, learning_rate_range, dropout_range,
                                         forecast, train_loader, val_loader, test_loader, criterion, device,
                                         max_epochs, patience, model_save_dir, model_name_LSTM,
                                         lookback, train_ratio, val_ratio, batch_size, random_seed,
                                         ImproveLSTM, Fv1_train_model_LSTM, Fv1_evaluate_model_on_test_set_LSTM,
                                         Fv1_evaluate_final_metrics_LSTM, Fv1_save_results_to_txt_LSTM,
                                         n_trials=20, timeout=3600, direction='minimize'):
    """
    使用 Optuna 进行超参数搜索的 LSTM 模型优化函数

    参数说明:
    ----------
    learning_rate_range : list
        学习率搜索范围，例如 [1e-2, 1e-3, 1e-4, 1e-5]

    dropout_range : list
        Dropout率搜索范围，例如 [0.2, 0.3, 0.4, 0.5]

    n_trials : int, default=20
        Optuna 试验次数（尝试的超参数组合数量）

    timeout : int, default=3600
        超参数搜索的超时时间（秒）

    direction : str, default='minimize'
        优化方向：'minimize'（最小化损失）或 'maximize'（最大化指标）

    返回:
    -------
    best_trial : optuna.trial.FrozenTrial
        最佳试验的详细信息

    study : optuna.study.Study
        完整的搜索研究，包含所有试验结果
    """

    print("开始使用 Optuna 进行 LSTM 超参数搜索...")
    print(f"层数范围: {num_layers_range}")
    print(f"隐藏单元范围: {hidden_size_range}")
    print(f"学习率范围: {learning_rate_range}")
    print(f"Dropout范围: {dropout_range}")
    print(f"计划试验次数: {n_trials}")
    print(f"超时时间: {timeout} 秒")

    # 创建 Optuna study
    study = optuna.create_study(
        direction=direction,                    # 优化方向
        sampler=optuna.samplers.TPESampler(),  # 使用 TPE 算法进行高效搜索
        pruner=optuna.pruners.HyperbandPruner()  # 使用 Hyperband 进行早停
    )

    # 定义带参数的 objective 函数
    objective_with_args = lambda trial: objective(
        trial, forecast, train_loader, val_loader, test_loader,
        criterion, device, max_epochs, patience, model_save_dir, model_name_LSTM,
        lookback, train_ratio, val_ratio, batch_size, random_seed, ImproveLSTM,
        Fv1_train_model_LSTM, Fv1_evaluate_model_on_test_set_LSTM,
        Fv1_evaluate_final_metrics_LSTM, Fv1_save_results_to_txt_LSTM,
        num_layers_range, hidden_size_range, learning_rate_range, dropout_range
    )

    # 执行超参数优化
    study.optimize(     # best_trial是Optuna优化过程中返回的最佳试验对象
        objective_with_args,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    # 输出最佳试验结果
    best_trial = study.best_trial
    print(f"\n*** Optuna 超参数搜索完成 ***")
    print(f"最佳试验: #{best_trial.number}")
    print(f"最佳超参数:")
    print(f"  - LSTM层数: {best_trial.params['num_layers']}")
    print(f"  - 隐藏单元: {best_trial.params['hidden_size']}")
    print(f"  - 学习率: {best_trial.params['learning_rate']:.6f}")
    print(f"  - Dropout率: {best_trial.params['dropout']:.2f}")
    print(f"最佳验证损失: {best_trial.value:.6f}")

    # 保存最佳模型配置和 Optuna 研究结果
    save_optuna_results(study, model_save_dir, best_trial)

    return best_trial, study


def save_optuna_results(study, model_save_dir, best_trial):
    """
    保存 Optuna 搜索结果
    """
    # 确保目录存在
    os.makedirs(model_save_dir, exist_ok=True)

    # 保存最佳超参数配置
    best_config_file = os.path.join(model_save_dir, "best_hyperparameters_optuna.txt")
    with open(best_config_file, 'w', encoding='utf-8') as f:
        f.write("Optuna 最佳超参数配置:\n")
        f.write(f"试验编号: {best_trial.number}\n")
        f.write(f"LSTM层数: {best_trial.params['num_layers']}\n")
        f.write(f"隐藏层单元数: {best_trial.params['hidden_size']}\n")
        f.write(f"学习率: {best_trial.params['learning_rate']:.0e}\n")
        f.write(f"Dropout率: {best_trial.params['dropout']:.2f}\n")
        f.write(f"最佳验证损失: {best_trial.value:.6f}\n")
        f.write(f"总试验次数: {len(study.trials)}\n\n")

        f.write("所有试验结果:\n")
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                f.write(f"试验 #{trial.number}: ")
                f.write(f"层数={trial.params['num_layers']}, ")
                f.write(f"隐藏单元={trial.params['hidden_size']}, ")
                f.write(f"学习率={trial.params['learning_rate']:.0e}, ")
                f.write(f"dropout={trial.params['dropout']:.2f}, ")
                f.write(f"损失={trial.value:.6f}\n")

    # 保存 Optuna 研究到文件（可用于后续分析）
    study_file = os.path.join(model_save_dir, "optuna_study.pkl")
    import joblib
    joblib.dump(study, study_file)

    print(f"Optuna 结果已保存至: {best_config_file}")
    print(f"完整研究已保存至: {study_file}")


def train_with_hyperparams(num_layers, hidden_size, learning_rate, dropout, forecast,
                           train_loader, val_loader, test_loader, criterion,
                           device, max_epochs, patience, model_save_dir,
                           model_name_LSTM, lookback, train_ratio, val_ratio,
                           batch_size, random_seed, ImproveLSTM,
                           Fv1_train_model_LSTM, Fv1_evaluate_model_on_test_set_LSTM,
                           Fv1_evaluate_final_metrics_LSTM, Fv1_save_results_to_txt_LSTM):
    """
    使用指定的超参数训练单个LSTM模型（支持学习率和dropout优化）
    """
    # 创建当前超参数对应的保存目录（包含学习率和dropout信息）
    current_model_save_dir = os.path.join(
        model_save_dir,
        f"{num_layers}layers_{hidden_size}units_lr{learning_rate:.0e}_do{dropout:.2f}"
    )
    os.makedirs(current_model_save_dir, exist_ok=True)

    # 根据预测步长选择模型结构
    if forecast == 1:
        model_instance = ImproveLSTM(1, hidden_size, num_layers, dropout, 1, 1)    # dropout在此处指定
        print(f"使用单步预测模型 - 层数: {num_layers}, 隐藏单元: {hidden_size}, "
              f"学习率: {learning_rate:.6f}, dropout: {dropout:.2f}")
    else:
        model_instance = ImproveLSTM(1, hidden_size, num_layers, dropout, 1, forecast)
        print(f"使用多步预测模型，预测步长: {forecast} - 层数: {num_layers}, "
              f"隐藏单元: {hidden_size}, 学习率: {learning_rate:.6f}, dropout: {dropout:.2f}")

    # 创建优化器（使用指定的学习率）
    optimizer = Adam(model_instance.parameters(), lr=learning_rate) # 模型训练时学习率在此指定

    # 调用训练函数
    trained_model, training_history = Fv1_train_model_LSTM(
        model_instance, train_loader, val_loader, criterion, optimizer,
        max_epochs, patience, device, current_model_save_dir,
        f"{model_name_LSTM}_{num_layers}layers_{hidden_size}units_lr{learning_rate:.0e}_do{dropout:.2f}"
    )

    # 在测试集上评估模型
    test_results = Fv1_evaluate_model_on_test_set_LSTM(trained_model, test_loader, criterion, device, forecast)

    # 计算训练集和验证集的最终指标
    train_metrics, val_metrics = Fv1_evaluate_final_metrics_LSTM(trained_model, train_loader, val_loader, criterion,
                                                                 device)

    # 准备模型配置信息
    model_config = {
        '预测步长 (forecast)': forecast,
        '历史步长 (lookback)': lookback,
        '隐藏层大小 (hidden_size)': hidden_size,
        'LSTM层数 (num_layers)': num_layers,
        '学习率 (learning_rate)': learning_rate,
        'Dropout率': dropout,
        '批大小 (batch_size)': batch_size,
        '训练集比例': train_ratio,
        '验证集比例': val_ratio,
        '测试集比例': 1 - train_ratio - val_ratio,
        '最大训练轮次': max_epochs,
        '早停轮数': patience,
        '随机种子': random_seed,
        '使用设备': str(device)
    }

    # 保存结果到txt文件
    results_file_path = os.path.join(current_model_save_dir,
                                     f"{model_name_LSTM}_{num_layers}layers_{hidden_size}units_lr{learning_rate:.0e}_do{dropout:.2f}_results.txt")
    Fv1_save_results_to_txt_LSTM(training_history, test_results, train_metrics, val_metrics, model_config,
                                 results_file_path)

    # 返回验证集上的最佳损失，用于超参数选择
    if training_history.get('best_val_loss'):   # 如果键存在且值不为 None/False/空
        best_val_loss = training_history['best_val_loss']   # 验证集上损失为1-R2，越小越好
    else:
        best_val_loss = float('inf')

    return best_val_loss, trained_model, training_history