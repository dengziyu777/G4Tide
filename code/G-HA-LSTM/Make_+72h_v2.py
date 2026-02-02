"""
改进LSTM潮位预报主程序
使用训练好的改进LSTM模型进行潮位预报
在主程序中指定模型参数
输入需要两个数据源：实测潮位和调和分析潮位
"""

import os
import warnings
warnings.filterwarnings('ignore')

# 导入自定义函数模块
import source_code.Fv1_functions4make as func

print("=" * 60)
print("改进LSTM潮位预报系统（主程序指定模型参数）")
print("=" * 60)

# ==================== 1. 设置参数 ====================
# 模型文件路径
model_path = './old_251227/output_ImproveLSTMv0_+720h/1layers_1024units_lr1e-03_do0.50/ImproveLSTMv0_1layers_1024units_lr1e-03_do0.50.pth'

# 输入数据文件路径（需要两个数据源）
obs_input_data_path = './input_make/O_YT_00.txt'  # 用户提供的实测潮位数据文件
ha_input_data_path = './input_make/HA_YT_00.txt'  # 用户提供的调和分析潮位数据文件

# 输出文件路径
output_path = './output_make/YT_00_forecast_results.csv'
gpu_id = 1                  # 训练时所调用的GPU编号，编号从0开始
lookback = 24               # 历史步长（默认24小时）

# 模型参数（在主程序中直接指定）
model_params = {
    'input_size': 1,        # 输入特征维度（历史实测数据+未来调和分析数据 合并输入模型，此处为1）
    'hidden_size': 1024,    # LSTM隐藏层单元数
    'num_layers': 1,        # LSTM层数
    'dropout': 0.5,         # Dropout率
    'output_size': 1,       # 输出维度
    'forecast_horizon': 720  # 预报步长
}

# 设备设置
import torch
# 指定GPU编号相关
if torch.cuda.is_available():   # 检查GPU是否可用
    torch.cuda.set_device(gpu_id)  # 设置GPU设备
    device = torch.device(f'cuda:{gpu_id}')
else:
    device = torch.device('cpu')
print(f"使用设备: {device}")

# ==================== 2. 加载模型 ====================
print(f"\n步骤1: 加载改进LSTM模型")
print("使用以下模型参数:")
for key, value in model_params.items():
    print(f"  {key}: {value}")

# 从.pth文件加载模型（不进行参数推断）
model = func.load_improve_model_with_given_params(
    model_path=model_path,
    device=device,
    model_params=model_params
)

if model is None:
    print("模型加载失败，程序退出")
    exit(1)

print(f"\n将使用以下参数进行预报:")
print(f"  预报步长: {model_params['forecast_horizon']}小时")
print(f"  历史步长: {lookback}小时")

# ==================== 3. 读取输入数据 ====================
print(f"\n步骤2: 读取输入数据")
print(f"实测数据文件: {obs_input_data_path}")
print(f"调和分析数据文件: {ha_input_data_path}")

# 读取两个数据源
obs_data = func.read_tide_data(obs_input_data_path)
ha_data = func.read_tide_data(ha_input_data_path)

if obs_data is None or ha_data is None:
    print("数据读取失败，程序退出")
    exit(1)

# 验证数据
is_valid_obs, message_obs = func.validate_tide_data(obs_data, min_length=lookback)
is_valid_ha, message_ha = func.validate_tide_data(ha_data, min_length=lookback+model_params['forecast_horizon'])

if not is_valid_obs:
    print(f"实测数据验证失败: {message_obs}")
    exit(1)
if not is_valid_ha:
    print(f"调和分析数据验证失败: {message_ha}")
    exit(1)

# 打印数据预览
print("\n实测潮位数据预览:")
func.print_data_preview(obs_data)
print("\n调和分析潮位数据预览:")
func.print_data_preview(ha_data)

# ==================== 4. 准备输入数据 ====================
print(f"\n步骤3: 准备输入数据")
input_tensor = func.prepare_input_data_for_improve_lstm(
    obs_data=obs_data,
    ha_data=ha_data,
    lookback=lookback,
    forecast_horizon=model_params['forecast_horizon']
)

if input_tensor is None:
    print("输入数据准备失败，程序退出")
    exit(1)

# ==================== 5. 生成预报 ====================
print(f"\n步骤4: 生成预报")
predictions = func.generate_forecast_for_improve(
    model=model,
    input_tensor=input_tensor,
    device=device,
    forecast_horizon=model_params['forecast_horizon']
)

if predictions is None:
    print("预报生成失败，程序退出")
    exit(1)

# ==================== 6. 保存结果 ====================
print(f"\n步骤5: 保存预报结果")
results_df = func.save_forecast_results(
    predictions=predictions,
    output_path=output_path
)

if results_df is None:
    print("结果保存失败，程序退出")
    exit(1)

# ==================== 7. 显示结果摘要 ====================
func.print_forecast_summary(predictions, model_params['forecast_horizon'])

print("\n" + "=" * 60)
print("预报完成")
print("=" * 60)