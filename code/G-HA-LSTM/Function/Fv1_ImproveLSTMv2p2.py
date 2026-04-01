import torch
import torch.nn as nn
import math


# %% 定义改进LSTM模型-适用输入输出设计（历史实测+未来初步预测 -> 未来实测）、未来不仅可用于潮位=================================================
class ImproveLSTM(nn.Module):
    """
    自注意力机制 + LSTM模型
    支持单步和多步预测
    改进：使用ResNet风格的残差连接
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 dropout=0.2, output_size=1, forecast_horizon=1,
                 use_attention=False, num_heads=1, use_resnet=True):
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
            num_heads: 自注意力头数（多头注意力）
            use_resnet: 是否使用ResNet风格的残差连接
        """
        super(ImproveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.use_resnet = use_resnet

        # 确保隐藏层大小能被头数整除
        assert hidden_size % num_heads == 0, "hidden_size必须能被num_heads整除"
        self.head_dim = hidden_size // num_heads

        # 输入投影层（如果需要调整维度）
        if input_size != hidden_size and use_resnet:
            self.input_projection = nn.Linear(input_size, hidden_size)
        else:
            self.input_projection = None

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=hidden_size if use_resnet and input_size != hidden_size else input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # 输入形状为(batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 自注意力机制（如果启用）
        if use_attention:
            # 线性变换层，用于生成Q、K、V
            self.W_q = nn.Linear(hidden_size, hidden_size)
            self.W_k = nn.Linear(hidden_size, hidden_size)
            self.W_v = nn.Linear(hidden_size, hidden_size)

            # 输出线性层
            self.fc_out = nn.Linear(hidden_size, hidden_size)

            # 层归一化
            self.layer_norm1 = nn.LayerNorm(hidden_size)
            self.layer_norm2 = nn.LayerNorm(hidden_size)

            # ResNet风格的前馈网络
            self.feed_forward = self.build_resnet_feed_forward(hidden_size, dropout) if use_resnet else \
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size * 4, hidden_size)
                )

        # 输出层 - 适配多步预测
        if forecast_horizon == 1:
            # 单步预测
            self.linear = nn.Linear(hidden_size, output_size)
        else:
            # 多步预测
            self.linear = nn.Linear(hidden_size, output_size * forecast_horizon)

        # 输出投影层（如果需要调整维度）
        if hidden_size != output_size * forecast_horizon and forecast_horizon > 1 and use_resnet:
            self.output_projection = nn.Linear(hidden_size, output_size * forecast_horizon)
        else:
            self.output_projection = None

    def build_resnet_feed_forward(self, hidden_size, dropout):
        """
        构建ResNet风格的前馈网络

        参数:
            hidden_size: 隐藏层大小
            dropout: dropout比率

        返回:
            ResNet风格的前馈网络
        """
        return ResNetBlock(
            in_channels=hidden_size,
            out_channels=hidden_size,
            expansion=4,  # 扩展因子
            dropout=dropout
        )

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力

        参数:
            Q: 查询张量 (batch_size, seq_len, hidden_size)
            K: 键张量 (batch_size, seq_len, hidden_size)
            V: 值张量 (batch_size, seq_len, hidden_size)
            mask: 注意力掩码

        返回:
            注意力输出和注意力权重
        """
        d_k = Q.size(-1)

        # 计算注意力分数: Q * K^T / sqrt(d_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # 应用掩码（如果有）
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax归一化得到注意力权重
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 注意力加权
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def multi_head_attention(self, x, mask=None):
        """
        多头注意力机制

        参数:
            x: 输入张量 (batch_size, seq_len, hidden_size)
            mask: 注意力掩码

        返回:
            多头注意力输出
        """
        batch_size, seq_len, _ = x.size()

        # 生成Q、K、V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 重塑为多头形状: (batch_size, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 转置以便批量计算: (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 计算缩放点积注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 转置回来并拼接: (batch_size, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )

        # 输出线性变换
        output = self.fc_out(attn_output)

        return output, attn_weights

    def forward(self, x, mask=None):
        """
        前向传播

        参数:
            x: 输入张量，形状为(batch_size, seq_len, input_size)
            mask: 注意力掩码，用于屏蔽无效位置

        返回:
            预测结果
        """
        batch_size = x.size(0)

        # 输入投影（如果需要）
        if self.input_projection is not None:
            x_proj = self.input_projection(x)
        else:
            x_proj = x

        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM前向传播
        # lstm_out形状: (batch_size, seq_len, hidden_size)
        lstm_out, (hn, cn) = self.lstm(x_proj, (h0, c0))

        # 添加LSTM残差连接
        if self.use_resnet:
            # 如果维度匹配，直接添加残差连接
            if x_proj.size(-1) == lstm_out.size(-1):
                lstm_out = lstm_out + x_proj
            # 否则通过1x1卷积调整维度
            elif hasattr(self, 'lstm_residual_proj'):
                residual = self.lstm_residual_proj(x_proj)
                lstm_out = lstm_out + residual

        # 自注意力机制
        if self.use_attention:
            # 残差连接前的原始输出
            residual = lstm_out

            # 多头注意力
            attn_output, attention_weights = self.multi_head_attention(lstm_out, mask)

            # Add & Norm (ResNet风格的残差连接)
            attn_output = self.dropout(attn_output)
            attn_output = self.layer_norm1(residual + attn_output)

            # 前馈网络（可能包含ResNet块）
            if self.use_resnet and hasattr(self.feed_forward, 'forward_resnet'):
                # 使用ResNet块
                ff_output = self.feed_forward(attn_output)
            else:
                # 标准前馈网络
                residual = attn_output
                ff_output = self.feed_forward(attn_output)
                ff_output = self.dropout(ff_output)
                attn_output = self.layer_norm2(residual + ff_output)
                ff_output = attn_output

            # 对序列维度取平均，得到上下文向量
            context = ff_output.mean(dim=1)
            last_output = context
        else:
            # 如果不使用注意力，仍然使用最后一个时间步的输出
            last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
            attention_weights = None

        # Dropout
        output = self.dropout(last_output)

        # 线性层
        if self.forecast_horizon == 1:
            # 单步预测
            output = self.linear(output)  # (batch_size, 1)
        else:
            # 多步预测
            if self.output_projection is not None and self.use_resnet:
                output = self.output_projection(output)
            else:
                output = self.linear(output)  # (batch_size, output_size * forecast_horizon)
            # 重塑为多步格式
            output = output.view(batch_size, self.forecast_horizon, self.output_size)
            # 如果是单变量，去掉最后一个维度
            if self.output_size == 1:
                output = output.squeeze(-1)  # (batch_size, forecast_horizon)

        return output

    def get_attention_weights(self, x, mask=None):
        """
        获取注意力权重（用于分析模型关注哪些时间步）

        参数:
            x: 输入张量
            mask: 注意力掩码

        返回:
            attention_weights: 注意力权重，形状(batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = x.size(0)

        # 输入投影（如果需要）
        if self.input_projection is not None:
            x_proj = self.input_projection(x)
        else:
            x_proj = x

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM前向传播
        lstm_out, (hn, cn) = self.lstm(x_proj, (h0, c0))

        # 计算注意力权重
        batch_size, seq_len, _ = lstm_out.size()

        # 生成Q、K
        Q = self.W_q(lstm_out)
        K = self.W_k(lstm_out)

        # 重塑为多头形状
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 转置以便批量计算
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用掩码（如果有）
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # 扩展到多头
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # 获取注意力权重
        attention_weights = torch.softmax(attention_scores, dim=-1)

        return attention_weights

    def init_hidden(self, batch_size, device):
        """初始化隐藏状态（可选方法）"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return h0, c0


class ResNetBlock(nn.Module):
    """
    ResNet风格的残差块
    包含两个线性层，每个线性层后接LayerNorm和激活函数
    """

    def __init__(self, in_channels, out_channels, expansion=4, dropout=0.2):
        """
        初始化ResNet块

        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            expansion: 扩展因子（隐藏层大小是输入/输出的多少倍）
            dropout: dropout比率
        """
        super(ResNetBlock, self).__init__()

        # 计算隐藏层大小
        hidden_channels = in_channels * expansion

        # 第一个线性层：in_channels -> hidden_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.activation1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        # 第二个线性层：hidden_channels -> out_channels
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.activation2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        # 如果输入和输出维度不同，需要投影
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.LayerNorm(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量

        返回:
            输出张量
        """
        residual = x

        # 第一个线性层
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.activation1(out)
        out = self.dropout1(out)

        # 第二个线性层
        out = self.fc2(out)
        out = self.norm2(out)

        # 残差连接
        out = out + self.shortcut(residual)

        # 最后的激活函数
        out = self.activation2(out)
        out = self.dropout2(out)

        return out


class ResNetLSTMBlock(nn.Module):
    """
    更复杂的ResNet-LSTM组合块
    包含LSTM层和残差连接
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2, bidirectional=False):
        """
        初始化ResNet-LSTM块

        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            dropout: dropout比率
            bidirectional: 是否双向
        """
        super(ResNetLSTMBlock, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 如果双向，输出维度加倍
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # LayerNorm
        self.layernorm = nn.LayerNorm(lstm_output_size)

        # 如果输入和输出维度不同，需要投影
        if input_size != lstm_output_size:
            self.shortcut = nn.Sequential(
                nn.Linear(input_size, lstm_output_size),
                nn.LayerNorm(lstm_output_size)
            )
        else:
            self.shortcut = nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        """
        前向传播

        参数:
            x: 输入张量 (batch_size, seq_len, input_size)
            hidden: 初始隐藏状态

        返回:
            lstm_out: LSTM输出
            (hn, cn): 最终隐藏状态
        """
        residual = x

        # LSTM前向传播
        lstm_out, (hn, cn) = self.lstm(x, hidden)

        # LayerNorm
        lstm_out = self.layernorm(lstm_out)

        # 残差连接
        lstm_out = lstm_out + self.shortcut(residual)

        # Dropout
        lstm_out = self.dropout(lstm_out)

        return lstm_out, (hn, cn)