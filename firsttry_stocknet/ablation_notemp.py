# 消融实验模型介绍：
# 仅使用结构相关性图（由大型语言模型生成）。
# 使用多层GCN处理结构相关性图，生成节点嵌入。
# 直接使用全连接层进行股价预测，不引入时间相关性图或注意力机制。
# 目的：验证结构相关性图对股价预测的贡献。
import torch
import torch.nn as nn
from layers import GraphConvolution
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def construct_graph(correlation_matrix):
    
    # 基于相关系数矩阵构建图，相关系数大于阈值的公司之间有边
    num_stocks = correlation_matrix.shape[0]
    edge_index = []
    edge_attr = []
    
    for i in range(num_stocks):
        for j in range(num_stocks):
            # 这里我们保留正负相关的边，只要它们的值不为0
            if correlation_matrix[i, j] != 0:  
                edge_index.append([i, j])
                edge_attr.append(correlation_matrix[i, j])  # 相关系数作为边权重
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_index, edge_attr

# 图卷积网络（GCN）
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)  # 第一层图卷积
        self.gc2 = GraphConvolution(hidden_dim, out_dim)    # 第二层图卷积
        self.dropout = dropout

    def forward(self, x, adj):
        # 通过两层图卷积提取节点特征
        x = F.relu(self.gc1(x, adj))  # 第一层图卷积 + ReLU激活
        x = F.dropout(x, self.dropout, training=self.training)  # Dropout
        x = self.gc2(x, adj)  # 第二层图卷积
        return x

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        return self.conv(x, edge_index, edge_attr)
 
# 图卷积网络（SFGCN）
class SFGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, dropout):
        super(SFGCN, self).__init__()

        # 定义三个GCN模块，分别处理结构图、特征图和公共图
        self.SGCN1 = GCN(input_dim, hidden_dim1, hidden_dim2, dropout)  # 处理结构图
        self.CGCN = GCN(input_dim, hidden_dim1, hidden_dim2, dropout)   # 处理公共图

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(hidden_dim2, 1)))  # 注意力参数
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化
        self.tanh = nn.Tanh()
        # 添加部分??
        self.gcn1 = GCNLayer(hidden_dim2, hidden_dim3) 
        self.gcn2 = GCNLayer(hidden_dim3, hidden_dim4)  
        self.fc=nn.Linear(hidden_dim4, 1)

    def forward(self, x, sadj, fadj, edge_index):
        # 通过结构图（sadj）和特征图（fadj）进行图卷积计算
        emb1 = self.SGCN1(x, sadj)  # Special_GCN1 -- 结构图
        emb = emb1
        
        # 使用全连接层进行股价预测
        x = self.gcn1(emb, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        # 使用全连接层进行股价预测
        y_pred = self.fc(x)
        
        # 返回图结构优化结果
        return y_pred, emb # 返回各个中间输出

