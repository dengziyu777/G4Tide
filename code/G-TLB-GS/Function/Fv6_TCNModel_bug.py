import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm as parametrize_weight_norm
weight_norm = parametrize_weight_norm



class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        """
        Temporal Convolutional Network (TCN) 用于多步预测；存在bug

        参数:
            input_size: 输入特征的维度（每时间步的特征数量）
            output_size: 输出序列长度（要预测的未来时间步数量）
            num_channels: 各层卷积的通道数列表，如[64,128]表示两层TCN，通道数分别为64和128
            kernel_size: 卷积核大小
            dropout: Dropout概率
        """
        super(TCNModel, self).__init__()
		# 保存模型参数
        self.input_size = input_size	# 输入特征维度
        self.output_size = output_size	# 输出时间步数量
        self.num_channels = num_channels	# 各层通道数列表
        self.kernel_size = kernel_size	# 卷积核尺寸
        self.dropout = dropout	# Dropout概率

        # 构建TCN层级结构
        layers = []
        num_levels = len(num_channels)	# TCN层级数
		
        for i in range(num_levels):
            dilation_size = 2 ** i	# 指数增长的膨胀率：1, 2, 4, 8...
			# 确定输入/输出通道数
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
			
			# 添加TCN基础块
            layers += [TCNBlock(in_channels, out_channels, kernel_size,
                                stride=1, dilation=dilation_size,
                                padding=(kernel_size - 1) * dilation_size,
                                dropout=dropout)]

        # 将所有TCN块组合成序列
		self.tcn = nn.Sequential(*layers)

        # 输出层：将最后一层输出投影到预测空间
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
		# 输入x形状: (batch_size, 序列长度, 特征维度)
		# 转置为卷积层要求的格式: (batch_size, 特征维度, 序列长度)
        x = x.transpose(1, 2) 
		
		# 通过TCN层堆叠
        out = self.tcn(x)	# 输出形状: (batch_size, 最后层通道数, 序列长度)
		
		# 取序列最后一个时间步的所有通道: (batch_size, 最后层通道数)
        out = self.linear(out[:, :, -1])
		
		# 通过线性层输出预测: (batch_size, 预测步长)
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