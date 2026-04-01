import torch
import torch.nn as nn


# %% 定义改进LSTM模型-适用输入输出设计（历史实测+未来初步预测 -> 未来实测）、未来不仅可用于潮位=================================================
class ImproveLSTM(nn.Module):
    """
    注意力机制（此为简单的加性注意力机制，使用v2p1中的自注意力机制）
    支持单步和多步预测
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 dropout=0.2, output_size=1, forecast_horizon=1,
                 use_attention=True):
        """
        初始化LSTM模型

        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层单元数
            num_layers: LSTM层数
            dropout: Dropout比率
            output_size: 输出维度
            forecast_horizon: 预测步长
            use_attention: 是否使用注意力机制
        """
        super(ImproveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size
        self.use_attention = use_attention

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # 输入形状为(batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 注意力机制（如果启用）
        if use_attention:
            # 简单注意力机制：计算每个时间步的重要性权重
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),  # 降维
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1)  # 输出单个注意力分数
            )

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
            x: 输入张量，形状为(batch_size, seq_len, 1)

        返回:
            预测结果
        """
        batch_size = x.size(0)

        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM前向传播
        # lstm_out形状: (batch_size, seq_len, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        # 注意力机制
        if self.use_attention:
            # 计算注意力权重
            # 1. 将每个时间步的隐藏状态输入到注意力网络
            attention_scores = self.attention(lstm_out)  # (batch_size, seq_len, 1)

            # 2. 使用softmax归一化，得到注意力权重
            attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)

            # 3. 加权求和得到上下文向量
            # context形状: (batch_size, hidden_size)
            context = torch.sum(attention_weights * lstm_out, dim=1)

            # 4. 使用上下文向量作为最终特征
            last_output = context
        else:
            # 如果不使用注意力，仍然使用最后一个时间步的输出
            last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)

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

    def get_attention_weights(self, x):
        """
        获取注意力权重（用于分析模型关注哪些时间步）

        参数:
            x: 输入张量

        返回:
            attention_weights: 注意力权重，形状(batch_size, seq_len)
        """
        batch_size = x.size(0)

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM前向传播
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))

        # 计算注意力权重
        attention_scores = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, seq_len, 1)

        return attention_weights.squeeze(-1)  # 去掉最后一个维度

    def init_hidden(self, batch_size, device):
        """初始化隐藏状态（可选方法）"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h0, c0