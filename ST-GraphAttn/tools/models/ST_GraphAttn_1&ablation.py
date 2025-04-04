import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
    
        # 确保两个张量的数据类型一致
        if isinstance(adj, np.ndarray):
            adj = torch.from_numpy(adj).float()  # 如果 adj 是 numpy.ndarray，则转换为 tensor
        support = support.to(torch.float32)  # 确保 support 张量也是 float32
        output = torch.mm(adj, support)

        if self.bias is not None:
            return (output + self.bias).to(device)
        else:
            return output.to(device)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
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
    
class ST_GraphAttn(nn.Module):
    def __init__(self, model_cfg):
        super(ST_GraphAttn, self).__init__()
        input_dim = model_cfg.INPUT_DIM
        hidden_dim1 = model_cfg.HIDDEN_DIM1
        hidden_dim2 = model_cfg.HIDDEN_DIM2
        hidden_dim3 = model_cfg.HIDDEN_DIM3
        hidden_dim4 = model_cfg.HIDDEN_DIM4
        lstm_hidden = model_cfg.LSTM_HIDDEN
        dropout = model_cfg.DROPOUT

        # Static Graph GCN
        self.SGCN = GCN(input_dim, hidden_dim1, hidden_dim2, dropout)
        # Dynamic Feature Graph GCN
        self.DGCN = GCN(input_dim, hidden_dim1, hidden_dim2, dropout)

        # Attention over multiple graph views
        self.attention = Attention(hidden_dim2)

        # Temporal modeling
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim2, batch_first=True)

        # Gated Fusion
        self.gate = nn.Linear(hidden_dim2 * 2, 1)

        # Spatial GCNConv + MLP
        self.gcn1 = GCNConv(hidden_dim2, hidden_dim3)
        self.gcn2 = GCNConv(hidden_dim3, hidden_dim4)
        self.out_fc = nn.Linear(hidden_dim4, 1)

    def forward(self, x, sadj, fadj, edge_index, **kwargs):
        """
        x: [N, T] -- 每个节点的时间序列
        sadj: [N, N] -- 静态邻接矩阵
        fadj: [N, N] -- 当前窗口内动态邻接矩阵
        edge_index: 图的边连接，供 GCNConv 使用
        """
        x_seq = x.unsqueeze(-1)
        N = x_seq.size(0)

        # Step 1: 时间序列建模
        lstm_out, _ = self.lstm(x_seq)           # [N, T, hidden_dim2]
        h_lstm = lstm_out[:, -1, :]              # [N, hidden_dim2]

        # Step 2: 图表示建模（静态结构图、动态特征图）
        x_current = x[:, -1]  # 取最后时间步的值作为当前输入 [N]
        x_current = x_current.unsqueeze(-1)  # [N, 1]

        gcn_static = self.SGCN(x_current, sadj)   # [N, hidden_dim2]
        gcn_dynamic = self.DGCN(x_current, fadj)  # [N, hidden_dim2]
        gcn_combined = (gcn_static + gcn_dynamic) / 2

        # Step 3: 多图融合
        multi_view = torch.stack([gcn_static, gcn_dynamic, gcn_combined], dim=1)
        gcn_feat, att = self.attention(multi_view)  # [N, hidden_dim2]

        # Step 4: Gated Fusion
        fusion_input = torch.cat([gcn_feat, h_lstm], dim=1)  # [N, hidden_dim2 * 2]
        z = torch.sigmoid(self.gate(fusion_input))  # [N, 1]
        
        # 使用广播机制进行融合
        fused = z * gcn_feat + (1 - z) * h_lstm  # [N, hidden_dim2]

        # Step 5: 空间图卷积预测
        x = self.gcn1(fused, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        out = self.out_fc(x)

        return out

# 消融实验
