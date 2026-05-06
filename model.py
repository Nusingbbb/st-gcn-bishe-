import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入你同级目录下的 graph.py，用于获取人体 22 个关节点的物理连结矩阵
from graph import Graph


# ==========================================
# 基础组件 1：空间图卷积层 (Spatial Graph Convolution)
# 负责在同一帧内，让相连的关节点互相交换位置信息
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
# 基础组件 2：时空图卷积块 (ST-GCN Block)
# 包含一层空间图卷积 + 一层时间卷积 (TCN) + 残差连接 (Residual)
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

        # 残差连接配置
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
        x = self.tcn(x) + res
        return self.relu(x), A


# ==========================================
# 核心网络：孪生时空特征提取器 (Siamese ST-GCN Feature Extractor)
# ==========================================
class STGCN_Regressor(nn.Module):
    def __init__(self, in_channels=3, edge_importance_weighting=True, **kwargs):
        super().__init__()

        # 1. 初始化图结构 (基于 UI-PRMD 的 22 节点图)
        self.graph = Graph()
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # 建立可学习的边权重矩阵 (Edge Importance)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        # 2. 搭建 9 层 ST-GCN 提取主干 (Backbone)
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

        # 是否开启边权重注意力机制
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # ==========================================
        # 🌟 2025顶会架构爆改点：全新的特征映射头 (MLP Projection Head) 🌟
        # ==========================================
        # 彻底抛弃分类用的 Linear(256, 10)，转而将 256 维的骨架特征
        # 进一步浓缩为 50 维的高级隐空间向量 (Feature Embedding)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 50)
        )

    def forward(self, x):
        # 输入维度解析：N(Batch), C(通道数, 通常为3), T(帧数, 73), V(关节点, 22), M(人数, 1)
        N, C, T, V, M = x.size()

        # 步骤 A：入口处数据归一化 (Data BatchNorm)
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # 步骤 B：特征逐层穿过 ST-GCN 大脑
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # 步骤 C：全局平均池化 (Global Average Pooling)
        # 消除时间和空间维度，提取出每个样本最纯粹的 256 维特征
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = x.view(N, -1)  # 此时形状为 (N, 256)

        # ==========================================
        # 🌟 输出层改动：特征发射 🌟
        # ==========================================
        # 将 256 维特征压入 MLP，生成最终的 50 维拓扑特征向量
        embedding = self.fc(x)

        # 返回形状为 (Batch, 50) 的特征矩阵，准备在 train.py 中进行余弦距离对决！
        return embedding