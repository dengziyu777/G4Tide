import torch
import torch.nn as nn


# %% 定义改进LSTM模型-适用输入输出设计（历史实测+未来初步预测 -> 未来实测）、未来不仅可用于潮位=================================================
class ImproveLSTM(nn.Module):
    """
    双向LSTM + 残差连接模型（用于潮位预测）
    支持单步和多步预测
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 dropout=0.2, output_size=1, forecast_horizon=1,
                 use_bidirectional=True, use_residual=True):
        """
        初始化双向残差LSTM模型

        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层单元数
            num_layers: LSTM层数
            dropout: Dropout比率
            output_size: 输出维度
            forecast_horizon: 预测步长
            use_bidirectional: 是否使用双向LSTM
            use_residual: 是否使用残差连接
        """
        super(ImproveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size
        self.use_bidirectional = use_bidirectional
        self.use_residual = use_residual

        # 双向LSTM的方向数
        self.num_directions = 2 if use_bidirectional else 1

        # 1. 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # 输入形状为(batch, seq, feature)
            bidirectional=use_bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        # 2. 残差连接层（可选）
        if use_residual:
            # 将输入维度映射到LSTM输出维度
            self.residual_linear = nn.Linear(
                input_size,
                hidden_size * self.num_directions
            )

        # 3. Dropout层
        self.dropout = nn.Dropout(dropout)

        # 4. 层归一化（可选，稳定训练）
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

        # 5. 特征转换层（增强特征表示）
        self.feature_transform = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        # 6. 输出层 - 适配多步预测
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
            x: 输入张量，形状为(batch_size, seq_len, input_size)

        返回:
            预测结果
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 1. 初始化双向LSTM的隐藏状态和细胞状态
        # 注意：双向LSTM需要2倍的隐藏状态
        h0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ).to(x.device)

        c0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ).to(x.device)

        # 2. 双向LSTM前向传播
        # lstm_out形状: (batch_size, seq_len, hidden_size * num_directions)
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        # 3. 只取最后一个时间步的输出
        # 对于双向LSTM，这包含了前向和后向的信息
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size * num_directions)

        # 4. 层归一化
        last_output = self.layer_norm(last_output)

        # 5. 残差连接（可选）
        if self.use_residual:
            # 取输入x的最后一个时间步
            last_input = x[:, -1, :]  # (batch_size, input_size)

            # 将输入映射到与LSTM输出相同的维度
            residual = self.residual_linear(last_input)  # (batch_size, hidden_size * num_directions)

            # 残差连接：LSTM输出 + 映射后的输入
            last_output = last_output + residual

        # 6. Dropout
        last_output = self.dropout(last_output)

        # 7. 特征转换
        transformed_output = self.feature_transform(last_output)

        # 8. 线性层输出
        if self.forecast_horizon == 1:
            # 单步预测
            output = self.linear(transformed_output)  # (batch_size, 1)
        else:
            # 多步预测
            output = self.linear(transformed_output)  # (batch_size, output_size * forecast_horizon)
            # 重塑为多步格式
            output = output.view(batch_size, self.forecast_horizon, self.output_size)
            # 如果是单变量，去掉最后一个维度
            if self.output_size == 1:
                output = output.squeeze(-1)  # (batch_size, forecast_horizon)

        return output

    def get_direction_info(self, x):
        """
        获取双向LSTM两个方向的信息（用于分析）

        参数:
            x: 输入张量

        返回:
            forward_info: 前向LSTM最后一个时间步的输出
            backward_info: 后向LSTM最后一个时间步的输出
        """
        if not self.use_bidirectional:
            return None, None

        batch_size = x.size(0)

        # 初始化隐藏状态
        h0 = torch.zeros(
            self.num_layers * 2,
            batch_size,
            self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers * 2,
            batch_size,
            self.hidden_size
        ).to(x.device)

        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))  # (batch, seq, hidden*2)

        # 分离前向和后向信息
        hidden_size = self.hidden_size
        forward_output = lstm_out[:, -1, :hidden_size]  # 前向
        backward_output = lstm_out[:, 0, hidden_size:]  # 后向（注意：后向第一个时间步对应原始序列最后一个）

        return forward_output, backward_output

    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        h0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ).to(device)

        c0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ).to(device)

        return h0, c0