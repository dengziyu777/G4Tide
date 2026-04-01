import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, bidirectional, dropout):
        super().__init__()
        self.input_size = input_size  # 保存input_size，用于验证模型时读取参数
        self.hidden_sizes = hidden_sizes  # 是一个列表
        self.num_layers = len(hidden_sizes)  # 层数由隐藏层大小列表的长度决定
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        direction_factor = 2 if bidirectional else 1  # 方向因子

        # 创建 LSTM 层列表
        self.lstm_layers = nn.ModuleList()

        # 计算每层的输入维度
        in_features = input_size
        for i, h_size in enumerate(hidden_sizes):
            # 添加 LSTM 层
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=in_features,
                    hidden_size=h_size,
                    num_layers=1,  # 每层只有一个LSTM单元
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=0  # 在层间手动添加dropout
                )
            )
            # 更新下一层的输入维度（考虑双向情况）
            in_features = h_size * direction_factor

            # 在非最后一层后添加 Dropout
            if i < len(hidden_sizes) - 1 and dropout > 0:
                self.lstm_layers.append(nn.Dropout(dropout))

        # 计算最后一层的输出维度
        final_out_features = hidden_sizes[-1] * direction_factor
        self.ln = nn.LayerNorm(final_out_features)  # 层归一化

        # 多层感知机（隐藏层大小可变）
        self.fc = nn.Sequential(
            nn.Linear(final_out_features, final_out_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_out_features // 2, output_size)
        )

    def forward(self, x):
        # 逐层处理 LSTM
        for layer in self.lstm_layers:
            if isinstance(layer, nn.LSTM):
                # LSTM 层返回 (output, (h_n, c_n))
                x, _ = layer(x)
            else:
                # Dropout 层直接处理
                x = layer(x)

        # 提取最后一个时间步的输出
        x = x[:, -1, :]
        x = self.ln(x)  # 归一化处理
        return self.fc(x)