import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm as parametrize_weight_norm
weight_norm = parametrize_weight_norm



class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        """
        Temporal Convolutional Network (TCN) 用于多步预测

        参数:
            input_size: 输入特征维度
            output_size: 输出序列长度（要预测的未来时间步数量）
            num_channels: 各层通道数，例如 [64, 128]
            kernel_size: 卷积核大小
            dropout: Dropout概率
        """
        super(TCNModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout

        # TCN层
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers += [TCNBlock(in_channels, out_channels, kernel_size,
                                stride=1, dilation=dilation_size,
                                padding=(kernel_size - 1) * dilation_size,
                                dropout=dropout)]

        self.tcn = nn.Sequential(*layers)

        # 输出层
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x形状: (batch_size, seq_len, input_size)
        x = x.transpose(1, 2)  # 转换为 (batch_size, input_size, seq_len)
        out = self.tcn(x)
        out = self.linear(out[:, :, -1])  # 取最后时间步
        return out


class TCNBlock(nn.Module):
    """TCN的基础块 (因果卷积 + 权重归一化 + ReLU + Dropout)"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TCNBlock, self).__init__()
        # 修改在这里：使用兼容性导入的weight_norm
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.padding = padding

    def forward(self, x):
        out = self.conv(x)
        # 移除多余的填充（右侧）
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        out = self.norm(out)
        out = self.relu(out)
        return self.dropout(out)