import numpy as np


class Graph():
    """
    定义 22 个人体关节点的图结构 (Graph) 和 邻接矩阵 (Adjacency Matrix)
    """

    def __init__(self, strategy='spatial'):
        # 你的数据集特征：22 个关节点
        self.num_node = 22
        self.strategy = strategy

        # 1. 获取物理连线
        self.get_edge()
        # 2. 生成数学矩阵
        self.get_adjacency()

    def get_edge(self):

        self.neighbor_link = [
            # 躯干连线 (假设 0是骨盆, 1是脊柱中, 2是胸, 3是脖子, 4是头)
            (0, 1), (1, 2), (2, 3), (3, 4),

            # 左臂连线 (假设 2连着左肩5, 5连左肘6, 6连左腕7, 7连左手8)
            (2, 5), (5, 6), (6, 7), (7, 8),

            # 右臂连线 (假设 2连着右肩9, 9连右肘10, 10连右腕11, 11连右手12)
            (2, 9), (9, 10), (10, 11), (11, 12),

            # 左腿连线 (假设 0连着左胯13, 13连左膝14, 14连左踝15, 15连左脚16)
            (0, 13), (13, 14), (14, 15), (15, 16),

            # 右腿连线 (假设 0连着右胯17, 17连右膝18, 18连右踝19, 19连右脚20)
            (0, 17), (17, 18), (18, 19), (19, 20),

            # 头部末端或其他延伸节点 (例如头顶 21)
            (4, 21)
        ]

        # 定义人体的物理重心节点（通常是骨盆/盆骨，这里假设为 0 号节点）
        self.center = 0

        # 自己连自己（保留节点自身的特征不丢失）
        self.self_link = [(i, i) for i in range(self.num_node)]

        # 所有的边 = 自身连线 + 物理连线
        self.edge = self.self_link + self.neighbor_link

    def get_adjacency(self):
        """
        将物理连线转化为神经网络能看懂的邻接矩阵 (Adjacency Matrix)
        """
        # 第一步：计算节点之间的最短距离 (用于判断向心/离心)
        hop_dis = self.get_hop_distance(self.num_node, self.edge)

        # 第二步：根据空间划分策略 (Spatial Configuration) 生成 3 个子矩阵
        if self.strategy == 'spatial':
            # 我们会生成一个形状为 (3, 22, 22) 的张量 A
            # 3 代表三种空间策略：根节点、向心节点(靠近重心)、离心节点(远离重心)
            A = np.zeros((3, self.num_node, self.num_node))

            for i in range(self.num_node):
                for j in range(self.num_node):
                    if hop_dis[i, j] == 0:
                        # 策略 1：节点自身 (i == j)
                        A[0, i, j] = 1
                    elif hop_dis[i, j] == 1:
                        # 只有相邻的节点 (距离为1) 才会有连接
                        # 策略 2：向心 (目标节点 j 比 i 更靠近身体中心)
                        if hop_dis[j, self.center] < hop_dis[i, self.center]:
                            A[1, i, j] = 1
                        # 策略 3：离心 (目标节点 j 比 i 更远离身体中心)
                        else:
                            A[2, i, j] = 1

            # 归一化矩阵（防止矩阵相乘时数值爆炸）
            self.A = self.normalize_adjacency(A)

        else:
            raise ValueError(f"不支持的策略: {self.strategy}")

    def get_hop_distance(self, num_node, edge):
        """
        计算图中任意两个节点之间的最短跳数 (Hop)
        """
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1

        hop_dis = np.zeros((num_node, num_node)) + np.inf
        # 自己到自己的距离是 0
        for i in range(num_node):
            hop_dis[i, i] = 0

        # 广度优先搜索 (BFS) 的矩阵乘法实现
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(num_node + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(num_node, -1, -1):
            hop_dis[arrive_mat[d]] = d

        return hop_dis

    def normalize_adjacency(self, A):
        """
        图卷积的标准数学操作：度归一化 (Degree Normalization)
        $$A_{norm} = D^{-1/2} * A * D^{-1/2}$$
        """
        Dl = np.sum(A, 0)
        num_node = A.shape[1]
        Dn = np.zeros((A.shape[0], num_node, num_node))
        for i in range(A.shape[0]):
            Dn[i] = np.diag(np.power(Dl[i] + 1e-8, -1))
        A_norm = np.matmul(Dn, A)
        return A_norm


# ==========================================
# 独立测试区
# ==========================================
if __name__ == '__main__':
    graph = Graph(strategy='spatial')
    A = graph.A
    print("✅ 图结构构建成功！")
    print(f"邻接矩阵 A 的形状: {A.shape}")
    # 预期输出: (3, 22, 22)。这 3 个通道将被送入 ST-GCN 进行空间图卷积计算。