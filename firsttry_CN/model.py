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

# 注意力机制
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        # 用线性层来计算注意力权重
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),  # 投影到hidden_size维度
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)  # 投影到1维进行注意力加权
        )

    def forward(self, z):
        # 计算注意力权重
        w = self.project(z)  
        beta = torch.softmax(w, dim=1)  # 使用softmax归一化权重
        # 计算加权和及返回注意力权重
        return (beta * z).sum(1), beta
    
# 添加的模型？?
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
        self.SGCN2 = GCN(input_dim, hidden_dim1, hidden_dim2, dropout)  # 处理特征图
        self.CGCN = GCN(input_dim, hidden_dim1, hidden_dim2, dropout)   # 处理公共图

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(hidden_dim2, 1)))  # 注意力参数
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化
        self.attention = Attention(hidden_dim2)  # 注意力机制
        self.tanh = nn.Tanh()
        # 添加部分??
        self.gcn1 = GCNLayer(hidden_dim2, hidden_dim3) 
        self.gcn2 = GCNLayer(hidden_dim3, hidden_dim4)  
        self.fc=nn.Linear(hidden_dim4, 1)

    def forward(self, x, sadj, fadj, edge_index):
        # 通过结构图（sadj）和特征图（fadj）进行图卷积计算
        emb1 = self.SGCN1(x, sadj)  # Special_GCN1 -- 结构图
        com1 = self.CGCN(x, sadj)   # Common_GCN -- 结构图
        com2 = self.CGCN(x, fadj)   # Common_GCN -- 特征图
        emb2 = self.SGCN2(x, fadj)  # Special_GCN2 -- 特征图

        # 融合图卷积结果（结构图卷积和特征图卷积结果加权平均）
        Xcom = (com1 + com2) / 2

        # 堆叠所有图卷积结果
        emb = torch.stack([emb1, emb2, Xcom], dim=1)
        
        # 使用注意力机制进行加权
        emb, att = self.attention(emb)  # 计算加权的节点表示及其注意力权重
        
        # 使用全连接层进行股价预测
        x = self.gcn1(emb, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        # 使用全连接层进行股价预测
        y_pred = self.fc(x)
        
        # 返回图结构优化结果
        return y_pred, emb1, com1, com2, emb2, emb # 返回各个中间输出

