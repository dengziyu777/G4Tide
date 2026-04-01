import os
import torch

# DZY自定义的函数
DEBUG_MODE = False  # Debug模式
if DEBUG_MODE:
    from source.Fv6_DebugTools import debug_print_meteo_data, debug_plot_interpolated_scatter
from source.Fv6_load_data_EL import Fv6_load_data_EL
from source.Fv6_load_meteo_data import Fv6_load_meteo_data
from source.Fv6_prepare_interpolated_data import Fv6_prepare_interpolated_data
from source.Fv6_prepare_predict_sequences_all_sites import Fv6_prepare_predict_sequences_all_sites
from source.Fv6_predict_NewF_ScaleXIndep import Fv6_predict_NewF_ScaleXIndep  # 导入预测函数
from source.Fv6_save_DL_P_to_dat import Fv6_save_DL_P_to_dat
from source.Fv6_data_preprocessing import Fv6_adjust_and_prepare_observation
from source.Fv6_adaptive_smoothing import Fv6_adaptive_smoothing

# DZY250707：利用LearnEL所得映射关系，输入预报潮位+环境场得新预报潮位-undone251215


# %% 1. 参数设置 ===========================================================================================
Observer_Data = False  # 是否有相应时段实测数据；若有需设置第5部分！

batch1_data_file_path = '.\A_TwoPoints\Make2016_input\MIKE生成潮位_2016010108toEnd_原数据+8h考虑为北京时间.dat'  # 天文潮数据
batch1_data_scale_factor = 1  # 乘以此数值将其转化为以m为单位
batch1_data_EL_Adjust = 0.0  # 基准面调整值（m）

batch3_data_folder_path = '.\A_TwoPoints\Make2016_input\era5-2016010108toEnd'  # 设置era5等网站下载的环境数据所在文件夹；ERA5中数据为UTC，转北京时间+8；站点顺序需保持一致
batch3_num_features = 10  # 每个站点的气象特征数

# 输入预报数据中DL预报时绘图与否------------------------
model_three_comparison = True  # 绘制三种数据比较效果与否
plot_common_period_only = True  # 图中仅绘制公共时段（绘图时显示历史时间步长）

# 利用DL模型预测时参数设置------------------------------------
use_forward_hours_all = 24  # 在此设置历史时间步长，免去繁琐操作；本脚本支持历史时间步任意、未来时间步长为0
batch2_data_time_interval_adjust = 3600  # 使用“DL模型”预测，时序数据时间间隔s（需要与model_folder_path中采用模型训练时设置一致，下同）

site_params_file = '.\A_TwoPoints\Make2016_input\模型参数设置-2016010300to123123.dat'
site_params = {}
with open(site_params_file, 'r') as f:
    for site_idx, line in enumerate(f):
        cols = line.strip().split()  # 解析每行数据，去除首尾空格并分割
        if len(cols) >= 5:  # 确保有5列数据：use_start_time, use_end_time, use_time_interval, use_forward_hours, use_backward_hours
            # 转换数据类型
            site_params[site_idx] = {
                "use_start_time": int(cols[0]), "use_end_time": int(cols[1]), "use_time_interval": int(cols[2]),
                "use_forward_hours": int(cols[3]), "use_backward_hours": int(cols[4])
            }
        else:
            raise ValueError(f"站点 {site_idx} 的参数数量不足，需要5个参数")

model_type = 'TCN'  # 模型类型选择：'TCN' 或 'LSTM'（以下需要与model_folder_path中采用模型训练时设置一致）
model_folder_path = f".\A_TwoPoints\Learn_output\-{use_forward_hours_all}h+0h_{model_type}-P4\model"  # 存放.pth模型的文件夹
batch_size = 2048  # 批尺寸
case_name = 'TwoPoints'  # 案例名称，用于本脚本寻找所使用的模型文件
case_name_1st = 'B'  # 本次预测案例首字母，绘图时站点编号用
interval_hours = 2400  # 绘图时,X轴的间隔（h）
rotation_user = 10  # 绘图时，X轴上时间的旋转角度（度）
y_scaling_method = 'forecast_based'  # Y标准化方法，global为基于全部观测数据统一标准化、forecast_based为基于各站点预测数据分别标准化

# 本次预测结果输出设置----------------------------------------
output_prediction_folder = f".\A_TwoPoints\Make2016_output\-{use_forward_hours_all}h+0h_{model_type}-P4"  # 便于自动创建输出目录
output_prediction_plot_folder = os.path.join(output_prediction_folder, 'plot')
output_evaluation_metrics_path = os.path.join(output_prediction_folder, f'{case_name}2DHD_EL_LEM.dat')
gpu_id = 0  # GPU编号
print("1.完成")

# %% 2.0 创建输出目录/设置所用GPU================================================================================================
os.makedirs(output_prediction_folder, exist_ok=True)
os.makedirs(output_prediction_plot_folder, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # 通过设置 CUDA_VISIBLE_DEVICES 环境变量，您可以指定程序可见的 GPU。程序将只看到您指定的 GPU。

# %% 2.1 读取输入层预报数据 ==========================================================================================
batch1_data = Fv6_load_data_EL(batch1_data_file_path, batch1_data_scale_factor, batch1_data_EL_Adjust,
                               False)  # 读取预报潮位数据
print(f"2.1完成")

# %% 2.2 读取输入层环境场（风、温度等）数据===============================================================================
batch3_data = Fv6_load_meteo_data(batch3_data_folder_path, batch3_num_features, False)
print(f"2.2完成\n")

# %% 3.1 调整输入层所用潮位、环境数据的起始时间、时间间隔====================================================================
sequence_data_per_site, timestamps_per_site = Fv6_prepare_interpolated_data(batch1_data, batch3_data, site_params,
                                                                            False)
print(f"3.1完成\n")

# %% 3.2 数据集划分==================================================================================================
site_sequences, predict_timestamps = Fv6_prepare_predict_sequences_all_sites(sequence_data_per_site,
                                                                             timestamps_per_site, site_params)
# for site_idx, seq in site_sequences.items():
#     print(f" 站点 {site_idx} 输入序列形状: {seq.shape} (输入层样本数×历史时间步×总特征数)")
print(f"3.2完成\n")

# %% 4 使用model_folder_path中训练好的模型进行预测 =================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备
predictions = {}  # 创建预测结果字典

# 对每个站点独立进行预测
for site_idx in site_sequences:
    # print(f"开始预测站点 {site_idx + 1}")
    site_seq = site_sequences[site_idx]  # 获取当前站点的输入序列

    site_prediction = Fv6_predict_NewF_ScaleXIndep(site_seq, model_folder_path, model_type, case_name, device,
                                                   batch_size, site_idx, y_scaling_method, False)  # 调用预测函数

    predictions[site_idx] = site_prediction  # 存储预测结果

print(f"4 所有站点预测完成\n")

# %% 5.输入层预报数据、输出层预报数据与实测数据绘图比较、输出================================================================
if DEBUG_MODE:
    from source.Fv6_three_comparison import Fv6_three_comparison

    batch2_data_file_path = '.\A_TwoPoints\Make2016_input\MIKE生成潮位_2016010108toEnd_原数据+8h考虑为北京时间.dat'  # 设置观测潮位数据所在文件夹（MQF师兄提供）（DL训练时，训练其数据覆盖时间）
    batch2_data_scale_factor = 1  # 乘以此数值将其转化为以m为单位
    batch2_data_EL_Adjust = 0.0  # 基准面调整值（m）

    batch2_data = Fv6_load_data_EL(batch2_data_file_path, batch2_data_scale_factor, batch2_data_EL_Adjust,
                                   False)  # 读取实测潮位数据
    batch2_data_adjusted = Fv6_adjust_and_prepare_observation(batch2_data, batch2_data_time_interval_adjust,
                                                              False)  # 调整实测数据的时间间隔

    Fv6_three_comparison(batch1_data, predictions, predict_timestamps, batch2_data_adjusted, site_params,
                         output_prediction_plot_folder,
                         f"F&DL_F&O / {case_name_1st}", f'EL(m)', interval_hours, rotation_user,
                         output_evaluation_metrics_path, plot_common_period_only,
                         batch3_data_folder_path)  # 保存未平滑对比图、评估结果
print(f"5 完成\n")

# %% 6.输出层预报数据输出=============================================================================================
print(f"DL预报结果未平滑...")
Fv6_save_DL_P_to_dat(predictions, predict_timestamps, output_prediction_folder, case_name)  # 保存DL预报结果-未平滑
print(f"6 完成")
