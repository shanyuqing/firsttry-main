import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import torch.optim as optim
from torch_geometric.nn import GATConv
import sys
import os
import numpy as np 
from scipy.stats import spearmanr  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
# GCN
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
    
# ST-GraphAttn
class ST_GraphAttn(nn.Module):
    def __init__(self, model_cfg):
        super(ST_GraphAttn, self).__init__()
        #input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, dropout
        input_dim = model_cfg.INPUT_DIM
        hidden_dim1 = model_cfg.HIDDEN_DIM1
        hidden_dim2 = model_cfg.HIDDEN_DIM2
        hidden_dim3 = model_cfg.HIDDEN_DIM3
        hidden_dim4 = model_cfg.HIDDEN_DIM4
        dropout = model_cfg.DROPOUT
        # 定义三个GCN模块，分别处理结构图、特征图和公共图
        self.SGCN1 = GCN(input_dim, hidden_dim1, hidden_dim2, dropout)  # 处理结构图
        self.SGCN2 = GCN(input_dim, hidden_dim1, hidden_dim2, dropout)  # 处理特征图
        self.CGCN = GCN(input_dim, hidden_dim1, hidden_dim2, dropout)   # 处理公共图

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(hidden_dim2, 1)))  # 注意力参数
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化
        self.attention = Attention(hidden_dim2)  # 注意力机制
        self.tanh = nn.Tanh()
        
        self.gcn1 = GCNConv(hidden_dim2, hidden_dim3) 
        self.gcn2 = GCNConv(hidden_dim3, hidden_dim4)  
        self.fc=nn.Linear(hidden_dim4, 1)

    def forward(self, x, sadj, fadj, edge_index, **kwargs):
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
        return y_pred # 返回各个中间输出

# GAT
class GAT(nn.Module):
    def __init__(self, model_cfg):
        super(GAT, self).__init__()
        input_size = model_cfg.INPUT_DIM
        hidden_size = model_cfg.HIDDEN_DIM
        output_size = model_cfg.OUTPUT_DIM
        num_heads = model_cfg.NUM_HEADS
        dropout = model_cfg.DROPOUT
        # 第一层GAT：多注意力头+拼接
        self.gat1 = GATConv(
            in_channels=input_size,
            out_channels=hidden_size,
            heads=num_heads,         # 使用num_heads个注意力头
            dropout=dropout,
            concat=True               # 默认True，拼接多头结果
        )
        
        # 层间Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 第二层GAT：单注意力头（输出层）+平均
        self.gat2 = GATConv(
            in_channels=hidden_size * num_heads,  # 输入维度是第一层的输出
            out_channels=output_size,
            heads=1,                # 输出层使用单头
            dropout=dropout,
            concat=False             # 不拼接（对多头结果取平均）
        )

    def forward(self, x, edge_index, sadj=None, fadj=None, **kwargs):
        # 第一层
        x = self.gat1(x, edge_index)
        x = F.elu(x)                 # 更常用的GAT激活函数
        x = self.dropout(x)
        
        # 第二层
        x = self.gat2(x, edge_index)
        return x  
    
# GRU_GAT
class GRU_GAT(nn.Module):
    def __init__(self, model_cfg):
        super(GRU_GAT, self).__init__()
        input_size = model_cfg.INPUT_DIM
        hidden_size = model_cfg.HIDDEN_DIM
        output_size = model_cfg.OUTPUT_DIM
        num_layers = model_cfg.NUM_LAYERS
        num_heads = model_cfg.NUM_HEADS
        dropout = model_cfg.DROPOUT
        
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # GAT layer
        self.gat1 = GATConv(hidden_size, hidden_size, num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_size * num_heads, output_size, heads=1, dropout=dropout)
        
    def forward(self, x, edge_index, sadj=None, fadj=None, **kwargs):
        # x: Node features (batch_size, seq_len, num_nodes, feature_dim)
        # edge_index: Graph connectivity (edge_index[0], edge_index[1])
        
        # First, process through GRU (Time-series part)
        x, _ = self.gru(x)  # x: (batch_size, seq_len, hidden_size)
       
        # GAT layer
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        
        return x


# GRU
class GRU(nn.Module):
    def __init__(self, model_cfg):
        super(GRU, self).__init__()
        input_size = model_cfg.INPUT_DIM
        hidden_size = model_cfg.HIDDEN_DIM
        output_size = model_cfg.OUTPUT_DIM
        num_layers = model_cfg.NUM_LAYERS
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, sadj=None, fadj=None, edge_index=None, **kwargs):
        out, _ = self.gru(x)
        # Take the output of the last time step
        out = self.fc(out)  
        return out
    
# LSTM_GAT
class LSTM_GAT(nn.Module):
    def __init__(self, model_cfg):
        super(LSTM_GAT, self).__init__()
        input_size = model_cfg.INPUT_DIM
        hidden_size = model_cfg.HIDDEN_DIM
        output_size = model_cfg.OUTPUT_DIM
        num_layers = model_cfg.NUM_LAYERS
        num_heads = model_cfg.NUM_HEADS
        dropout = model_cfg.DROPOUT
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # GAT layer
        self.gat1 = GATConv(hidden_size, hidden_size, heads=num_heads, dropout=0.5)
        self.gat2 = GATConv(hidden_size * num_heads, output_size, heads=1, dropout=0.5)
        
    def forward(self, x, edge_index,  sadj=None, fadj=None, **kwargs):
        # x: Node features (batch_size, seq_len, num_nodes, feature_dim)
        # edge_index: Graph connectivity (edge_index[0], edge_index[1])
        
        # First, process through LSTM (Time-series part)
        x, _ = self.lstm(x)  
        
        # GAT layer
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        
        return x
    
# LSTM
class LSTM(nn.Module):
    def __init__(self, model_cfg):
        super(LSTM, self).__init__()
        input_size = model_cfg.INPUT_DIM
        hidden_size = model_cfg.HIDDEN_DIM
        output_size = model_cfg.OUTPUT_DIM
        num_layers = model_cfg.NUM_LAYERS
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, sadj=None, fadj=None, edge_index=None, **kwargs):
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = self.fc(out)  
        return out
    
# RNN
class RNN(nn.Module):
    def __init__(self, model_cfg):
        super(RNN, self).__init__()
        input_size = model_cfg.INPUT_DIM
        hidden_size = model_cfg.HIDDEN_DIM
        output_size = model_cfg.OUTPUT_DIM
        num_layers = model_cfg.NUM_LAYERS
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, sadj=None, fadj=None, edge_index=None, **kwargs):
        out, _ = self.rnn(x)
        # Take the output of the last time step
        out = self.fc(out) 
        return out