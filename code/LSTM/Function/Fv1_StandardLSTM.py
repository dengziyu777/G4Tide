import torch
import torch.nn as nn

# %% 1. 定义标准的LSTM模型=====================================================================
class StandardLSTM(nn.Module):
    """
    标准LSTM模型（用于潮位）
    支持单步和多步预测
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 dropout=0.2, output_size=1, forecast_horizon=1):
        """
        初始化LSTM模型

        参数:
            input_size: 输入特征维度（本文标准LSTM用于潮位时，输入特征为1）
            hidden_size: 隐藏层单元数
            num_layers: LSTM层数
            dropout: Dropout比率
            output_size: 输出维度（单变量为1）
            forecast_horizon: 预测步长
        """
        super(StandardLSTM, self).__init__()
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
            x: 输入张量，形状为(batch_size, lookback, 1)

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
        # lstm_out形状: (batch_size, lookback, hidden_size)
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



# %% 2. 针对标准LSTM模型进行debug=====================================================================
def test_model_creation_LSTM(hidden_size, num_layers, dropout):
    """测试LSTM模型创建和基本前向传播
    使用用户配置的超参数

    参数:
        hidden_size: 隐藏层单元数
        num_layers: LSTM层数
        dropout: Dropout率
    """

    print("测试LSTM模型...")

    # 测试单步预测
    model_single = StandardLSTM(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_size=1,
        forecast_horizon=1
    )

    # 测试多步预测
    forecast = 12   # 未来时间步数
    model_multi = StandardLSTM(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_size=1,
        forecast_horizon=forecast  # 未来时间步数
    )

    # 测试前向传播
    batch_size = 32

    # 创建测试输入
    lookback = 24   # 历史时间步数
    x_test = torch.randn(batch_size, lookback, 1)

    # 单步预测测试
    with torch.no_grad():
        output_single = model_single(x_test)
        output_multi = model_multi(x_test)

    print(f"输入形状: {x_test.shape}")
    print(f"单步预测输出形状: {output_single.shape}")
    print(f"多步预测输出形状: {output_multi.shape}")

    # 验证输出形状是否符合预期
    expected_single = (batch_size, 1)
    expected_multi = (batch_size, forecast)

    assert output_single.shape == expected_single, f"单步预测形状错误: {output_single.shape} != {expected_single}"
    assert output_multi.shape == expected_multi, f"多步预测形状错误: {output_multi.shape} != {expected_multi}"

    print("标准LSTM模型测试通过!")
    return model_single, model_multi