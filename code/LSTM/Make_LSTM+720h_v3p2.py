"""
LSTM潮位预报主程序
使用训练好的模型进行720小时潮位预报
无需标准化操作
在主程序中指定模型参数
"""

import os
import warnings
warnings.filterwarnings('ignore')

# 导入自定义函数模块
import source_code.Fv1p2_functions4make as func

print("=" * 60)
print("LSTMv3潮位预报系统（主程序指定模型参数）")
print("=" * 60)

# ==================== 1. 设置参数 ====================
# 模型文件路径
model_path = './old_251227/output_LSTM_+720h/trial0_1layers_1024units_lr1e-03_do0.50/LSTM_trial0_1layers_1024units_lr1e-03_do0.50.pth'
input_data_path = './input_make/O_YT_00.txt'  # 用户提供的潮位数据文件
output_path = './output_make/O_YT_00_forecast_results.csv'
gpu_id = 1                  # 训练时所调用的GPU编号，编号从0开始
lookback = 24               # 历史步长（默认24小时）

# 模型参数（在主程序中直接指定）
model_params = {
    'input_size': 1,        # 输入特征维度（潮位为1）
    'hidden_size': 1024,    # LSTM隐藏层单元数
    'num_layers': 1,        # LSTM层数
    'dropout': 0.5,         # Dropout率
    'output_size': 1,       # 输出维度
    'forecast_horizon': 720 # 预报步长
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
print(f"\n步骤1: 加载模型")
print("使用以下模型参数:")
for key, value in model_params.items():
    print(f"  {key}: {value}")

# 从.pth文件加载模型（不进行参数推断）
model = func.load_model_with_given_params(
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
print(f"输入文件: {input_data_path}")

tide_data = func.read_tide_data(input_data_path)

if tide_data is None:
    print("数据读取失败，程序退出")
    exit(1)

# 验证数据
is_valid, message = func.validate_tide_data(tide_data, min_length=lookback)
if not is_valid:
    print(f"数据验证失败: {message}")
    exit(1)

# 打印数据预览
func.print_data_preview(tide_data)

# ==================== 4. 准备输入数据 ====================
print(f"\n步骤3: 准备输入数据")
input_tensor, original_data = func.prepare_input_data(
    tide_data=tide_data,
    lookback=lookback
)

if input_tensor is None:
    print("输入数据准备失败，程序退出")
    exit(1)

# ==================== 5. 生成预报 ====================
print(f"\n步骤4: 生成预报")
predictions = func.generate_forecast(
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