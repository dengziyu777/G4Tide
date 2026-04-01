"""
改进LSTM潮位预报功能函数模块
基于改进LSTMv2p1模型，输入包含两个数据源
支持主程序指定模型参数
输入特征为1（潮位值）
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
from torch import nn
warnings.filterwarnings('ignore')

# ==================== 改进LSTM模型定义 ====================

class ImproveLSTM(nn.Module):
    """
    改进LSTM模型（用于潮位）
    支持单步和多步预测
    输入特征数为1（潮位值），与训练时一致
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 dropout=0.2, output_size=1, forecast_horizon=1):
        """
        初始化改进LSTM模型

        参数:
            input_size: 输入特征维度（潮位为1）
            hidden_size: 隐藏层单元数
            num_layers: LSTM层数
            dropout: Dropout比率
            output_size: 输出维度（单变量为1）
            forecast_horizon: 预测步长
        """
        super(ImproveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # 输入形状为(batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0  # 只有多层LSTM时才使用dropout
        )

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 输出层 - 适配多步预测
        if forecast_horizon == 1:
            # 单步预测
            self.linear = nn.Linear(hidden_size, output_size)
        else:
            # 多步预测
            self.linear = nn.Linear(hidden_size, output_size * forecast_horizon)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状为(batch_size, lookback+forecast_horizon, 1)

        返回:
            预测结果，形状为:
            - 单步预测: (batch_size, 1)
            - 多步预测: (batch_size, forecast_horizon)
        """
        batch_size = x.size(0)

        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM前向传播
        # lstm_out形状: (batch_size, lookback+forecast_horizon, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        # 只取最后一个时间步的输出
        # last_output形状: (batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]

        # Dropout
        output = self.dropout(last_output)

        # 线性层
        if self.forecast_horizon == 1:
            # 单步预测
            output = self.linear(output)  # (batch_size, 1)
        else:
            # 多步预测
            output = self.linear(output)  # (batch_size, output_size * forecast_horizon)
            # 重塑为多步格式
            output = output.view(batch_size, self.forecast_horizon, self.output_size)
            # 如果是单变量，去掉最后一个维度
            if self.output_size == 1:
                output = output.squeeze(-1)  # (batch_size, forecast_horizon)

        return output

    def init_hidden(self, batch_size, device):
        """初始化隐藏状态（可选方法）"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h0, c0

# ==================== 数据读取相关函数 ====================

def read_tide_data(file_path):
    """
    从文本文件读取潮位数据

    参数:
        file_path: 文本文件路径

    返回:
        tide_data: 潮位数据列表
    """
    try:
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在")
            return None

        # 读取文本文件
        tide_data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            if line:  # 跳过空行
                try:
                    # 转换为浮点数
                    value = float(line)
                    tide_data.append(value)
                except ValueError:
                    print(f"警告: 第{i+1}行无法解析为数值: {line}")

        if not tide_data:
            print("错误: 文件中没有有效数据")
            return None

        print(f"从 {file_path} 成功读取 {len(tide_data)} 个潮位数据")
        return tide_data

    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return None

def validate_tide_data(tide_data, min_length=24, max_length=1000):
    """
    验证潮位数据

    参数:
        tide_data: 潮位数据列表
        min_length: 最小长度
        max_length: 最大长度

    返回:
        is_valid: 是否有效
        message: 验证信息
    """
    if tide_data is None:
        return False, "数据为空"

    if not isinstance(tide_data, (list, np.ndarray)):
        return False, "数据格式不正确，应为列表或数组"

    if len(tide_data) < min_length:
        return False, f"数据长度不足，需要至少{min_length}个数据点，当前只有{len(tide_data)}个"

    if len(tide_data) > max_length:
        return False, f"数据长度过长，最多允许{max_length}个数据点，当前有{len(tide_data)}个"

    # 检查是否为数值
    for i, value in enumerate(tide_data):
        try:
            float(value)
        except (ValueError, TypeError):
            return False, f"第{i+1}个数据不是有效的数值: {value}"

    return True, "数据验证通过"

def print_data_preview(tide_data, num_preview=10):
    """
    打印数据预览

    参数:
        tide_data: 潮位数据列表
        num_preview: 预览数量
    """
    if tide_data is None or len(tide_data) == 0:
        print("没有数据可预览")
        return

    print(f"\n数据预览 (共{len(tide_data)}个数据点):")
    print("-" * 40)

    # 打印前几个数据
    print(f"前{min(num_preview, len(tide_data))}个数据:")
    for i in range(min(num_preview, len(tide_data))):
        print(f"  [{i+1:3d}] {tide_data[i]:.4f}")

    if len(tide_data) > num_preview * 2:
        print("  ...")
        # 打印最后几个数据
        print(f"最后{num_preview}个数据:")
        for i in range(len(tide_data) - num_preview, len(tide_data)):
            print(f"  [{i+1:3d}] {tide_data[i]:.4f}")

    # 打印统计信息
    tide_array = np.array(tide_data, dtype=np.float32)
    print(f"\n数据统计:")
    print(f"  最小值: {tide_array.min():.4f}")
    print(f"  最大值: {tide_array.max():.4f}")
    print(f"  平均值: {tide_array.mean():.4f}")
    print(f"  标准差: {tide_array.std():.4f}")

# ==================== 模型加载相关函数 ====================

def load_improve_model_with_given_params(model_path, device, model_params):
    """
    使用给定的模型参数加载已训练的改进LSTM模型
    不从文件名或检查点推断参数，完全使用传入的参数

    参数:
        model_path: 模型文件路径
        device: 计算设备
        model_params: 模型参数字典，包含以下键:
            - input_size: 输入特征维度（应为1）
            - hidden_size: 隐藏层大小
            - num_layers: LSTM层数
            - dropout: dropout率
            - output_size: 输出维度
            - forecast_horizon: 预测步长

    返回:
        model: 加载的模型
    """
    print(f"正在从 {model_path} 加载改进LSTM模型...")
    print("使用以下模型参数:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")

    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在")
        return None

    try:
        # 检查模型参数是否完整
        required_params = ['input_size', 'hidden_size', 'num_layers',
                          'dropout', 'output_size', 'forecast_horizon']

        for param in required_params:
            if param not in model_params:
                print(f"错误: 缺少必需的模型参数: {param}")
                return None

        # 从参数字典中提取值
        input_size = model_params['input_size']
        hidden_size = model_params['hidden_size']
        num_layers = model_params['num_layers']
        dropout = model_params['dropout']
        output_size = model_params['output_size']
        forecast_horizon = model_params['forecast_horizon']

        # 验证输入特征数（改进LSTM应为1，与训练时一致）
        if input_size != 1:
            print(f"警告: 改进LSTM的input_size应为1，当前为{input_size}")

        # 创建改进LSTM模型实例
        model = ImproveLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size,
            forecast_horizon=forecast_horizon
        )

        # 加载检查点
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 检查点包含模型状态字典
            model_state_dict = checkpoint['model_state_dict']
        else:
            # 检查点本身就是模型状态字典
            model_state_dict = checkpoint

        # 加载模型权重
        model.load_state_dict(model_state_dict)

        # 将模型移动到指定设备
        model.to(device)
        model.eval()  # 设置为评估模式

        print(f"改进LSTM模型加载成功")

        return model

    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return None

def prepare_input_data_for_improve_lstm(obs_data, ha_data, lookback, forecast_horizon):
    """
    准备输入数据用于改进LSTM预测（不进行标准化）
    按照训练时的数据格式：历史实测潮位 + 未来HA潮位

    参数:
        obs_data: 实测潮位数据，至少包含lookback个值
        ha_data: 调和分析潮位数据，至少包含lookback+forecast_horizon个值
        lookback: 历史步长
        forecast_horizon: 预测步长

    返回:
        input_tensor: 准备好的输入张量，形状为(1, lookback+forecast_horizon, 1)
    """
    # 将输入转换为numpy数组
    if isinstance(obs_data, list):
        obs_data_array = np.array(obs_data, dtype=np.float32)
    elif isinstance(obs_data, np.ndarray):
        obs_data_array = obs_data.astype(np.float32)
    else:
        print(f"不支持的数据类型: {type(obs_data)}")
        return None

    if isinstance(ha_data, list):
        ha_data_array = np.array(ha_data, dtype=np.float32)
    elif isinstance(ha_data, np.ndarray):
        ha_data_array = ha_data.astype(np.float32)
    else:
        print(f"不支持的数据类型: {type(ha_data)}")
        return None

    # 检查数据长度是否足够
    if len(obs_data_array) < lookback:
        print(f"实测数据长度({len(obs_data_array)})小于所需的历史步长({lookback})")
        return None

    if len(ha_data_array) < lookback + forecast_horizon:
        print(f"调和分析数据长度({len(ha_data_array)})小于所需的历史步长({lookback}) + 预测步长({forecast_horizon})")
        return None

    # 取实测数据的最后lookback个值作为历史实测潮位
    hist_obs = obs_data_array[-lookback:].reshape(-1, 1)

    # 取HA数据的接下来的forecast_horizon个值作为未来HA潮位
    # 假设实测数据和HA数据是时间对齐的
    ha_start_idx = len(obs_data_array) - lookback
    future_ha = ha_data_array[ha_start_idx + lookback:ha_start_idx + lookback + forecast_horizon].reshape(-1, 1)

    # 合并历史实测和未来HA
    # 形状: (lookback+forecast_horizon, 1)
    combined_array = np.vstack([hist_obs, future_ha])

    # 注意：不进行标准化操作
    print("注意：输入数据不进行标准化操作")
    print(f"输入数据构建:")
    print(f"  历史实测数据: 取最后{lookback}个值，形状: {hist_obs.shape}")
    print(f"  未来HA数据: 取接下来{forecast_horizon}个值，形状: {future_ha.shape}")
    print(f"  合并后数据形状: {combined_array.shape}")

    # 转换为PyTorch张量
    input_tensor = torch.FloatTensor(combined_array).unsqueeze(0)  # 添加batch维度

    print(f"输入数据准备完成")
    print(f"  最终输入形状: {input_tensor.shape}")
    print(f"  历史实测范围: {hist_obs.min():.4f} ~ {hist_obs.max():.4f}")
    print(f"  未来HA范围: {future_ha.min():.4f} ~ {future_ha.max():.4f}")

    return input_tensor

def generate_forecast_for_improve(model, input_tensor, device, forecast_horizon):
    """
    生成预报（不进行反标准化）

    参数:
        model: 训练好的模型
        input_tensor: 输入张量 (1, lookback+forecast_horizon, 1)
        device: 计算设备
        forecast_horizon: 预报步数

    返回:
        predictions: 预报结果数组
    """
    print(f"正在生成预报...")

    model.eval()

    with torch.no_grad():
        # 将数据移到设备
        input_tensor = input_tensor.to(device)

        # 生成预报
        try:
            predictions = model(input_tensor)
            # 移回CPU并转换为numpy
            predictions = predictions.cpu().numpy().flatten()
        except Exception as e:
            print(f"预报生成失败: {str(e)}")
            return None

    print(f"注意：预报结果不进行反标准化操作")
    print(f"成功生成 {len(predictions)} 个预报值")
    return predictions

def save_forecast_results(predictions, output_path):
    """
    保存预报结果到文件

    参数:
        predictions: 预报值数组
        output_path: 输出文件路径

    返回:
        results_df: 包含预报结果的DataFrame
    """
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"创建输出目录: {output_dir}")

        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'forecast_hour': [(i+1) for i in range(len(predictions))],
            'predicted_value': predictions
        })

        # 保存到CSV文件
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"预报结果已保存到: {output_path}")
        print(f"  文件包含 {len(predictions)} 行预报数据")

        return results_df

    except Exception as e:
        print(f"保存预报结果失败: {str(e)}")
        return None

def print_forecast_summary(predictions, forecast_horizon):
    """
    打印预报结果摘要

    参数:
        predictions: 预报结果数组
        forecast_horizon: 预报步数
    """
    if predictions is None or len(predictions) == 0:
        print("没有预报结果可显示")
        return

    print("\n" + "=" * 60)
    print("预报结果统计:")
    print("=" * 60)
    print(f"预报总时长: {forecast_horizon} 小时 ({forecast_horizon/24:.1f} 天)")
    print(f"预报最小值: {np.min(predictions):.4f}")
    print(f"预报最大值: {np.max(predictions):.4f}")
    print(f"预报平均值: {np.mean(predictions):.4f}")
    print(f"预报标准差: {np.std(predictions):.4f}")

    print(f"\n预报结果预览 (前20小时):")
    print(f"{'小时':<8} {'预报值':<12}")
    print("-" * 20)
    for i, pred in enumerate(predictions[:20], 1):
        print(f"{i:<8} {pred:<12.4f}")

    if len(predictions) > 20:
        print(f"... 共 {len(predictions)} 个预报值")