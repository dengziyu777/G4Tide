import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna   # 开源的调参工具
from tqdm import tqdm
from torch.optim import Adam
import warnings
warnings.filterwarnings('ignore')

# DZY自定义的函数
DEBUG_MODE = True  # Debug模式（仅DZY251202编写此脚本时使用）
from source_code.Fv1_load_data_EL import Fv1_load_data_EL
from source_code.Fv1p2_preprocess_data_ImproveLSTM import Fv1_preprocess_data_ImproveLSTM
from source_code.Fv1_create_data_loaders_LSTM import Fv1_create_data_loaders_LSTM
from source_code.Fv1_train_model_LSTM import Fv1_train_model_LSTM
from source_code.Fv1_evaluate_model_on_test_set_LSTM import Fv1_evaluate_model_on_test_set_LSTM
from source_code.Fv1_evaluate_final_metrics_LSTM import Fv1_evaluate_final_metrics_LSTM
from source_code.Fv1_save_results_to_txt_LSTM import Fv1_save_results_to_txt_LSTM
from source_code.Fv1p2_optuna_hyperparameter_tuning_LSTM import Fv1_hyperparameter_search_optuna_LSTM
from source_code.Fv1_ImproveLSTMv2p1 import ImproveLSTM     # 尝试不同改进LSTM时，调整此处即可

# 251205 使用实测潮位数据，基于改进LSTM01预报（改进深度学习模型）------------

# %% 1.1. 设置本次训练参数/超参数====================================================================
forecast = 72   # 未来时间步长（72 240 360 720）
lookback = 24       # 历史时间步长（时间单位与batch1_data_file_path中数据一致）
train_ratio = 0.7       # 训练集比例
val_ratio = 0.15        # 验证集比例
batch_size = 64         # 批次大小
max_epochs = 2000         # 最大训练轮数
patience = 20            # 早停轮数
gpu_id = 0              # 训练时所调用的GPU编号，编号从0开始
random_seed = 251202  # 随机种子（250517 251202 251226）

# 用户可配置的LSTM超参数（使用Optuna进行超参数寻优）
num_layers_range = [1]      # LSTM层数范围（隐藏层一般为1-3层）
hidden_size_range = [1024]    # LSTM隐藏层单元数范围（24h 48h 7d 15d 30d，设置比历史时间步数稍小的隐藏层单元）
learning_rate_range = [1e-3]      # 学习率
dropout_range = [0.50]          # Dropout率

# Optuna 搜索参数
n_trials = 1           # Optuna 试验次数（要运行的试验次数）
timeout = 259200          # 超时时间（超参数搜索的最大时间（以秒为单位））

model_save_dir = './output_ImproveLSTMv2p1_+' + str(forecast) + 'h'     # 输出目录
model_name_LSTM = 'ImproveLSTMv2p1'   # 模型名称

batch1_data_file_path = '.\input\O_LHTandYT_81to00.txt'  # 设置实测潮位数据所在文件夹
batch1_data_scale_factor = 1        # 乘以此数值将其转化为以m为单位
batch1_data_EL_Adjust = 0.0     # 基准面调整值（m）

batch2_data_file_path = '.\input\HA_LHTandYT_81to00.txt'  # 设置HA潮位数据所在文件夹（时间起始需要与batch1一致）
batch2_data_scale_factor = 1        # 乘以此数值将其转化为以m为单位
batch2_data_EL_Adjust = 0.0     # 基准面调整值（m）

print(f"1.1. 完成")

# %% 1.2. 设置损失函数、随机种子、所用GPU=====================================================================
# 定义、使用R²损失函数相关
def r2_score_loss(y_true, y_pred):
    # R² = 1 - (SS_res / SS_tot)
    # 损失函数 = 1 - R² = SS_res / SS_tot
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)  # 添加小值防止除零
    return 1 - r2  # 返回1-R²作为损失（越小越好）

criterion = r2_score_loss   # 使用1-R²作为损失函数

# 随机种子相关
torch.manual_seed(random_seed)
np.random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)

# 指定GPU编号相关
if torch.cuda.is_available():   # 检查GPU是否可用
    torch.cuda.set_device(gpu_id)  # 设置GPU设备
    device = torch.device(f'cuda:{gpu_id}')
else:
    device = torch.device('cpu')

print(f"1.2. 完成")

# %% 2.1. 读取实测、调和分析潮位数据=====================================================================
batch1_data = Fv1_load_data_EL(batch1_data_file_path,batch1_data_scale_factor,batch1_data_EL_Adjust,False)      # batch1_data两列数据：时间戳、实测潮位
batch2_data = Fv1_load_data_EL(batch2_data_file_path,batch2_data_scale_factor,batch2_data_EL_Adjust,False)      # batch1_data两列数据：时间戳、调和分析潮位
print(f"2.1. 完成")

# %% 2.2. 预处理2.1中数据，构造数据集=====================================================================
preprocessed_data = Fv1_preprocess_data_ImproveLSTM(batch1_data, batch2_data, lookback, forecast, train_ratio, val_ratio)
print(f"2.2. 完成")

# %% 3.1. 创建数据加载器=====================================================================
train_loader, val_loader, test_loader = Fv1_create_data_loaders_LSTM(preprocessed_data, batch_size)
print(f"3.1. 完成")

# %% 3.2. 执行超参数搜索=====================================================================
best_trial, study = Fv1_hyperparameter_search_optuna_LSTM(
    num_layers_range=num_layers_range,
    hidden_size_range=hidden_size_range,
    learning_rate_range=learning_rate_range,  # 学习率范围
    dropout_range=dropout_range,              # dropout范围
    forecast=forecast,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    criterion=criterion,
    device=device,
    max_epochs=max_epochs,
    patience=patience,
    model_save_dir=model_save_dir,
    model_name_LSTM=model_name_LSTM,
    lookback=lookback,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    batch_size=batch_size,
    random_seed=random_seed,
    ImproveLSTM=ImproveLSTM,
    Fv1_train_model_LSTM=Fv1_train_model_LSTM,
    Fv1_evaluate_model_on_test_set_LSTM=Fv1_evaluate_model_on_test_set_LSTM,
    Fv1_evaluate_final_metrics_LSTM=Fv1_evaluate_final_metrics_LSTM,
    Fv1_save_results_to_txt_LSTM=Fv1_save_results_to_txt_LSTM,
    n_trials=n_trials,
    timeout=timeout,
    direction='minimize'
)

if best_trial and best_trial.state == optuna.trial.TrialState.COMPLETE:
    print(f"\n*** Optuna 超参数搜索完成 ***")
    print(f"最佳超参数: ")
    print(f"  - LSTM层数: {best_trial.params['num_layers']}")
    print(f"  - 隐藏单元: {best_trial.params['hidden_size']}")
    print(f"  - 学习率: {best_trial.params['learning_rate']:.6f}")
    print(f"  - Dropout率: {best_trial.params['dropout']:.2f}")
    print(f"最佳验证损失: {best_trial.value:.6f}")

print(f"3.2. 完成")