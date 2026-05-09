import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入图结构
from graph import Graph


# ==========================================
# 🌟 新增核心：骨骼时空注意力模块 (ST-CBAM) 🌟
# ==========================================
class ST_Attention_Module(nn.Module):
    """
    专为时空图卷积设计的轻量级注意力模块。
    包含 通道注意力(Channel) 和 时空注意力(Spatial-Temporal)。
    """

    def __init__(self, channels, reduction=4):
        super(ST_Attention_Module, self).__init__()

        # 1. 通道注意力：关注哪些特征维度最能反映动作错误
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

        # 2. 时空注意力：在 T(帧) 和 V(关节) 的维度上画“重点”
        # 使用 7x7 的大卷积核，感受野能覆盖多个相连的关节和连续的帧
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t, v = x.size()

        # --- 第一步：通道注意力 (Channel Attention) ---
        # 提取每个通道的全局平均特征
        y = self.avg_pool(x).view(b, c)
        # 计算每个通道的权重 (0~1)
        y = self.channel_excitation(y).view(b, c, 1, 1)
        # 将权重乘回原特征
        x_ca = x * y.expand_as(x)

        # --- 第二步：时空注意力 (Spatial-Temporal Attention) ---
        # 在通道维度上进行池化，找出在(t, v)坐标上响应最强烈的地方
        avg_out = torch.mean(x_ca, dim=1, keepdim=True)
        max_out, _ = torch.max(x_ca, dim=1, keepdim=True)

        # 将平均响应和最大响应拼接，交给卷积层去学习时空热力图
        spatial_feat = torch.cat([avg_out, max_out], dim=1)  # Shape: (b, 2, t, v)
        spatial_weight = self.spatial_sigmoid(self.spatial_conv(spatial_feat))  # Shape: (b, 1, t, v)

        # 最终输出 = 经过通道提纯的特征 * 时空重要性权重
        return x_ca * spatial_weight


# ==========================================
# 基础组件 1：空间图卷积层 (Spatial Graph Convolution)
# ==========================================
class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1,
                 bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A


# ==========================================
# 基础组件 2：时空图卷积块 (包含 Attention)
# ==========================================
class st_gcn_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        # 🌟 植入注意力模块
        self.attention = ST_Attention_Module(channels=out_channels)

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x)

        # 🌟 在残差相加之前，通过注意力机制去噪并高亮关键特征！
        x = self.attention(x)

        x = x + res
        return self.relu(x), A


# ==========================================
# 核心网络：孪生时空注意力特征提取器 (Siamese AST-GCN)
# ==========================================
class STGCN_Regressor(nn.Module):
    def __init__(self, in_channels=3, edge_importance_weighting=True, **kwargs):
        super().__init__()

        self.graph = Graph()
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        # 9 层带注意力的残差网络堆叠
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 128, kernel_size, 2, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 256, kernel_size, 2, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
        ))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # 投影头 (Projection Head)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 50)
        )

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # 特征穿过图卷积大脑 (期间经过 9 次时空注意力提纯)
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = x.view(N, -1)

        # 映射至高级隐空间
        embedding = self.fc(x)

        return embedding