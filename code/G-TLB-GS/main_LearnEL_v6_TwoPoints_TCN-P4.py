import os
import numpy as np
import torch
import random
from joblib import load
import time  # 导入时间模块

# DZY自定义的函数
DEBUG_MODE = False  # Debug模式
if DEBUG_MODE:
    from source.Fv6_DebugTools import debug_print_meteo_data, debug_plot_interpolated_scatter
output_debug_folder = '.\A_TwoPoints\debug'       # 该文件夹需存在
from source.Fv6_load_data_EL import Fv6_load_data_EL
from source.Fv6_load_meteo_data import Fv6_load_meteo_data
from source.Fv6_validate_and_plot import Fv6_validate_and_plot
from source.Fv6_evaluate_forecast import Fv6_evaluate_forecast
from source.Fv6_data_preprocessing import Fv6_adjust_and_prepare_observation, Fv6_extract_overlap_period
from source.Fv6_align_and_evaluate import Fv6_align_and_evaluate
from source.Fv6_write_evaluation_metrics import Fv6_write_evaluation_metrics_part6
from source.Fv6_prepare_sequence_data_with_meteo import Fv6_prepare_sequence_data_with_meteo
from source.Fv6_create_dataloaders import Fv6_create_dataloaders
from source.Fv6_TCNModel import TCNModel
from source.Fv6_LSTMModel import LSTMModel
from source.Fv6_train_UseAllModel import Fv6_train_UseAllModel
from source.Fv6_evaluate_model_per_site import Fv6_evaluate_model_per_site
from source.Fv6_SHAP_analysis_per_site import Fv6_SHAP_analysis_per_site
from source.write_runtime_statistics import write_runtime_statistics

# 251205:本脚本用于提升“区域水动力模型”模拟效果


# %% 1.1 设置本次训练参数====================================================================
batch1_data_file_path = '.\A_TwoPoints\Learn_input\input_F_MIKE\F_TwoPoints_81to00_+0h.dat'  # 调和分析数据
batch1_data_scale_factor = 1  # 乘以此数值将其转化为以m为单位
batch1_data_EL_Adjust = 0.0  # 基准面调整值（m）

batch2_data_file_path = '.\A_TwoPoints\Learn_input\input_O_BHSea\O_TwoPoints_81to00.dat'  # 实测数据
batch2_data_scale_factor = 0.01        # 乘以此数值将其转化为以m为单位
batch2_data_EL_Adjust = 0.0     # 基准面调整值（m）

batch3_data_folder_path = '.\A_TwoPoints\Learn_input\input_era5'  # 设置era5等网站下载的环境数据所在文件夹；ERA5中数据为UTC，转北京时间+8；站点顺序需保持一致
batch3_num_features = 10    # 每个站点的气象特征数

# “3.1”“4.2”，尤其是“6”中数据集划分和标准化 参数
batch2_data_time_interval_adjust = 3600  # 基于给定的观测数据，生成以此(s)为间隔的时序数据，插值方案选择
Fv5_FandO_plot_ON = False                # 3.1中绘制对比图时，是否仅绘制公共时段
forward_hours = 24                # 历史时间步长（使用向前use_forward_hoursh数据，0表示仅利用当前数据）
forward_steps_4SHAP = 5                 # SHAP分析时选择的代表性历史时间步数量（一旦历史时间步长不为0，此处需要大于1）
backward_hours = 0              # 未来时间步长（向后预测多少h数据，0表示仅预测当前）；训练时采用
backward_hours_4SHAP = 1        # SHAP分析时选择的代表性未来时间步数量（一旦历史时间步长不为0，此处需要大于1）
train_ratio = 0.7                       # 训练集比例
val_ratio = 0.15                         # 验证集比例
y_scaling_method = 'forecast_based'     # Y标准化方法，global为基于全部观测数据统一标准化、forecast_based为基于各站点预测数据分别标准化

case_name = 'TwoPoints'  # 本次预测案例名称
case_name_1st = 'B'     # 本次预测案例首字母，绘图时站点编号用
interval_hours = 2400  # 绘图时,X轴的间隔（h）
rotation_user = 10 # 绘图时，X轴上时间的旋转角度（度）
output_source_PandO_folder = f".\A_TwoPoints\Learn_output\-{forward_hours}h+{backward_hours}h_TCN-P4"  # 设置未优化预测、实测潮位的对比图输出文件夹
output_evaluation_metrics = os.path.join(output_source_PandO_folder, f'{case_name}2DHD_LEM.txt')  # 训练指标输出至该文件夹；LEM：Learn_Evaluation_Metrics
output_model_pth = os.path.join(output_source_PandO_folder, 'model')# 模型的pth、pkl文件输出至此
gpu_id = 1  # 训练时所调用的GPU编号，编号从0开始
# shap分析的设置
input_feature_names = ['AT', 'u10','v10','d2m','t2m', 'msl', 'sp', 'e','sro','ssro','tp']   # 特征名称，首先是batch1_data_folder_path对应特征，而后依次是batch3_data_folder_path对应特征
output_shap_path = os.path.join(output_source_PandO_folder, 'SHAP')# 模型的pth、pkl文件输出至此
SHAP_ANALYSIS_ENABLED = True  # 如果启用SHAP分析设置为True
SHAP_N_BACKGROUND = 500  # SHAP背景样本数
SHAP_MAX_SAMPLES = 500  # SHAP分析使用的最大样本数

# “7”、“8”和“9”模型训练、测试、验证时参数设置================================
### ------------------------ 通用参数 ------------------------
model_type = 'TCN'      # 模型类型选择：'TCN' 或 'LSTM'
batch_size = 2048          # 批尺寸
model_dropout = 0.2     # Dropout概率（防止过拟合，随机丢弃神经元比例）
max_epochs = 2000        # 最大训练轮数
patience = 200           # 早停等待轮数
min_delta = 0.001        # 视为改进的最小变化量（以R2来比较）
random_seed = 250517    # 随机种子；'6'中数据集划分用；据说42是个神奇的数字
### ------------------------ TCN模型专用参数 ------------------------
if model_type == 'TCN':
    tcn_kernel_size = 3                             # 控制感受野大小（如3、5、7等），影响捕捉时序依赖的范围。
    tcn_channels =[256, 256, 256, 256, 128, 128, 128, 128]          # 每层卷积的输出通道数（如64、128），决定特征提取的复杂度。
    print(f" TCN参数: 核大小={tcn_kernel_size}, 通道={tcn_channels}")

### ------------------------ LSTM模型专用参数 ------------------------
elif model_type == 'LSTM':
    lstm_layers_size = [512, 256, 128]              # 隐藏层层数（1-3）、每层神经元数量（32-512，常见128-256）
    lstm_bidirectional = False         # 是否使用双向LSTM（增加参数量但提高精度）；同时处理过去和未来信息，提高精度但增加计算量
    print(f" LSTM参数: 隐藏层大小={lstm_layers_size}, 层数={len(lstm_layers_size)}, 双向={lstm_bidirectional}")
print(f"1.1参数设置完成，模型类型: {model_type}")


# %% 1.2 依据随机种子控制全局=====================================================================
# 在程序开始处添加全局计时器
start_time = time.time()
torch.manual_seed(random_seed)      # 设置PyTorch的CPU随机数生成器种子
torch.cuda.manual_seed_all(random_seed)     # 设置所有GPU的随机种子
np.random.seed(random_seed)     # 设置NumPy的随机种子
random.seed(random_seed)    # 设置Python内置随机模块的种子
if model_type == 'LSTM':
    torch.backends.cudnn.deterministic = True  # 强制cuDNN使用确定性算法（TCN模型训练时采用确定性的极慢）
print(f"1.2完成")


# %% 2.0 设置输出目录、训练时所用GPU=====================================================================
os.makedirs(output_source_PandO_folder, exist_ok=True)  # 确保输出目录存在
# 指定GPU编号相关
if torch.cuda.is_available():   # 检查GPU是否可用
    torch.cuda.set_device(gpu_id)  # 设置GPU设备
    device = torch.device(f'cuda:{gpu_id}')
else:
    device = torch.device('cpu')
print(f"2.0完成")

# %% 2.1 读取输入层预报数据、输出层实测数据=====================================================================
batch1_data = Fv6_load_data_EL(batch1_data_file_path,batch1_data_scale_factor,batch1_data_EL_Adjust,False)  # 读取预报潮位数据
batch2_data = Fv6_load_data_EL(batch2_data_file_path,batch2_data_scale_factor,batch2_data_EL_Adjust,False)  # 读取实测潮位数据
print(f"2.1完成")

# %% 2.2 读取输入层气象场（风、温度等）数据=====================================================================
batch3_data = Fv6_load_meteo_data(batch3_data_folder_path,  batch3_num_features,False)
print(f"2.2完成\n")


# %% 3.1 预报、实测数据验证和可视化===============================================================
# Fv6_validate_and_plot(batch1_data, batch2_data, output_source_PandO_folder, f'F&O - {case_name_1st} - ',batch3_data_folder_path,f'EL(m)',
#                       Fv5_FandO_plot_ON,forward_hours, interval_hours,rotation_user)
# print(f"3.1完成\n")

# %% 3.2 评估原始预报数据与实测数据（仅取观测值存在的时刻）=================================================
# 评估原始预报数据
Fv6_evaluate_forecast(batch1_data, batch2_data, output_evaluation_metrics, f"3.2原始预报数据与实测数据评估结果（取公共时段、未插值观测数据）", False)
print(f"3.2完成\n")


# %% 4.1 调整实测数据的时间间隔，以期生成更多的训练数据==============================================
batch2_data_adjusted = Fv6_adjust_and_prepare_observation(batch2_data, batch2_data_time_interval_adjust,False)
print(f"4.1完成\n")

# %% 4.2 提取重叠时间段数据（考虑预测数据提前）======================================================================
fc_overlap_list, obs_overlap_list = Fv6_extract_overlap_period(batch1_data, batch2_data_adjusted, forward_hours,False)
print(f"4.2完成\n")


# %% 5. 各站点潮位数据对齐与评估====================================================================================
#full为包含提前时刻的，模型训练用；comon为与观测同时段的，评价指标用
X_full_list, t_full_list, X_common_list, Y_common_list, t_common_list = Fv6_align_and_evaluate(fc_overlap_list, obs_overlap_list,forward_hours,
    batch2_data_time_interval_adjust, output_evaluation_metrics, output_debug_folder,DEBUG_MODE,False)

# 写入详细的评估结果
Fv6_write_evaluation_metrics_part6(X_common_list, Y_common_list, output_evaluation_metrics,
    note=f"5.原始预报数据与实测数据评估结果（取公共时段、观测数据时间步长插值为{batch2_data_time_interval_adjust}s）")
print(f"5.完成")


# %% 6. 数据集划分和标准化（按站点独立处理）=========================================================================
num_sites = len(X_full_list)  # 获取站点数量，列数为站点数量
combined_data = Fv6_prepare_sequence_data_with_meteo(X_full_list, t_full_list, Y_common_list, t_common_list,
    batch3_data, batch2_data_time_interval_adjust, forward_hours, backward_hours,
    train_ratio, val_ratio, output_model_pth, model_type, case_name,y_scaling_method,False)
print(f"6.数据集划分和标准化完成\n")


# %% 7. 准备数据加载器（使用各站点合并后数据）===============================================================================
train_loader = Fv6_create_dataloaders(combined_data['X_train'],combined_data['Y_train'],batch_size)
val_loader = Fv6_create_dataloaders(combined_data['X_val'],combined_data['Y_val'],batch_size)
test_loader = Fv6_create_dataloaders(combined_data['X_test'],combined_data['Y_test'],batch_size)
print(f"训练集/验证集/测试集加载器: {len(train_loader)} // {len(val_loader)} // {len(test_loader)}批次")
print(f"7.准备数据加载器完成\n")


# %% 8 多站点预测模型训练（使用TCN、LSTM等）===============================================================================
# 获取输入输出特性
num_features = combined_data['num_features']
backward_length = combined_data['backward_length']    # 要预测的未来时间步数量

# 根据模型类型选择创建模型
if model_type == 'TCN':
    model = TCNModel(num_features, backward_length, tcn_channels, tcn_kernel_size, model_dropout)
    print(f"  -> 创建TCN模型，通道: {tcn_channels}，核大小: {tcn_kernel_size}")
elif model_type == 'LSTM':
    model = LSTMModel(num_features, lstm_layers_size, backward_length, lstm_bidirectional,model_dropout)
    print(f"  -> 创建LSTM模型，隐藏层: {lstm_layers_size}，层数: {len(lstm_layers_size)}，双向: {lstm_bidirectional}")
else:
    raise ImportError(f"未知模型类型: {model_type}")

# 调用新的训练函数（适用于所有站点合并训练）
Fv6_train_UseAllModel(model, model_type, train_loader, val_loader, output_model_pth,
                      case_name, 10, max_epochs, patience, min_delta)
print("8.模型训练 完成\n")


# %% 9. 评估多站点预测模型，测试集评估 ========================================================
# 执行分站点评估
site_metrics = Fv6_evaluate_model_per_site(combined_data, output_model_pth, output_evaluation_metrics, model_type,
    case_name,batch2_data_time_interval_adjust)

print("9.模型评估完成")

# 记录第10部分之前的时间
before_shap_time = time.time()

# %% 10. SHAP特征贡献度分析========================================================================
# SHAP分析是一个计算密集型过程，建议仅在小数据集上进行
# 注意：此步骤可能需要很长时间（可能超过模型训练时间）
# 因此设置为可选，通过`SHAP_ANALYSIS_ENABLED`控制是否执行

if SHAP_ANALYSIS_ENABLED:
    # 指定要分析的输入层、输出层时间步索引
    analysis_input_steps = "auto"
    analysis_output_step = "auto"

    # 获取所有站点
    total_sites = len(combined_data['site_datasets'])

    # 遍历每个站点进行分析
    for site_idx in range(total_sites):
        print(f"\n{'=' * 40} 站点 {site_idx + 1} SHAP分析开始 {'=' * 40}")

        # 获取当前站点数据
        site_data = combined_data['site_datasets'][site_idx]
        feature_scaler_path = os.path.join(output_model_pth,f"{model_type}_{case_name}_scaler_X{site_idx + 1}.pkl")

        try:
            # 加载当前站点的特征标准化器
            feature_scaler = load(feature_scaler_path)
            print(f"成功加载站点 {site_idx + 1} 的特征标准化器")
        except Exception as e:
            print(f"加载站点 {site_idx + 1} 的特征标准化器失败: {str(e)}")
            continue

        # 创建本站点的训练数据加载器（用于背景数据）
        train_loader_site = Fv6_create_dataloaders(site_data['X_train'], site_data['Y_train'],
            batch_size=min(len(site_data['X_train']), SHAP_N_BACKGROUND))

        # 创建本站点的测试数据加载器（用于解释数据）
        test_loader_site = Fv6_create_dataloaders(site_data['X_test'], site_data['Y_test'],
            batch_size=min(len(site_data['X_test']), SHAP_MAX_SAMPLES))

        # 运行SHAP分析
        Fv6_SHAP_analysis_per_site(model_type, output_model_pth, case_name,
            train_loader_site, test_loader_site, input_feature_names, site_idx, # 编号从0开始
            feature_scaler, SHAP_N_BACKGROUND, SHAP_MAX_SAMPLES, output_shap_path,
            analysis_input_steps, analysis_output_step,
            forward_hours, batch2_data_time_interval_adjust, forward_steps_4SHAP,
            backward_hours, backward_hours_4SHAP,batch3_data_folder_path)

    # 记录程序结束时间
    end_time = time.time()
    # 调用时间统计函数
    write_runtime_statistics(start_time,before_shap_time,end_time,SHAP_ANALYSIS_ENABLED,output_source_PandO_folder,gpu_id)

    print("10.SHAP特征贡献度分析完成")

