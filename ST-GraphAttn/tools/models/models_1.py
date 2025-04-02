import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import SpecialSpmm
from torch_geometric.nn import GATConv

class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, model_cfg):
        """Sparse GAT for regression tasks."""
        super(GAT, self).__init__()
        self.dropout = model_cfg.DROPOUT

        self.attentions = [SpGraphAttentionLayer(
            model_cfg.INPUT_DIM, model_cfg.HIDDEN_DIM, dropout=model_cfg.DROPOUT, alpha=model_cfg.ALPHA, concat=True)
            for _ in range(model_cfg.NUM_HEADS)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(
            model_cfg.HIDDEN_DIM * model_cfg.NUM_HEADS, model_cfg.OUTPUT_DIM, dropout=model_cfg.DROPOUT, alpha=model_cfg.ALPHA, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)  # no ELU for regression (optional)
        return x  # no log_softmax

# model = GAT(nfeat=1433, nhid=8, noutput=1, dropout=0.6, alpha=0.2, nheads=8)
# criterion = nn.MSELoss()
# output = model(features, adj) adj为邻接矩阵
# loss = criterion(output, labels)


class LSTMCellOriginal(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCellOriginal, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x_t, state):
        h_prev, c_prev = state
        combined = torch.cat((x_t, h_prev), dim=1)

        i_t = self.sigmoid(self.W_i(combined))         # Input gate
        o_t = self.sigmoid(self.W_o(combined))         # Output gate
        c_hat = self.tanh(self.W_c(combined))          # Candidate cell input

        c_t = c_prev + i_t * c_hat                     # No forget gate (original paper)
        h_t = o_t * self.tanh(c_t)

        return h_t, c_t

# 输入x时，需要对x进行扩展
# x = x.unsqueeze(-1)  例如从 (100, 20) → (100, 20, 1)
# 需要将INPUT_DIM改为1
class LSTM(nn.Module):
    def __init__(self, model_cfg):
        super(LSTM, self).__init__()
        self.hidden_size = model_cfg.HIDDEN_DIM
        self.lstm_cell = LSTMCellOriginal(model_cfg.INPUT_DIM, model_cfg.HIDDEN_DIM)
        self.fc = nn.Linear(model_cfg.HIDDEN_DIM, model_cfg.OUTPUT_DIM)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            h_t, c_t = self.lstm_cell(x[:, t, :], (h_t, c_t))

        output = self.fc(h_t)  # predict from final hidden state
        return output

# 使用范例
# model = LSTM(input_size=1, hidden_size=64)  # 只用收盘价时 input_size=1
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# # 输入：batch_size x 20天 x 每天特征维度（比如1维收盘价）
# x = torch.randn(32, 20, 1)      # 模拟32支股票的历史数据
# y = torch.randn(32, 1)          # 第21天股价

# output = model(x)
# loss = criterion(output, y)
# loss.backward()
# optimizer.step()


class LSTM_GAT(nn.Module):
    def __init__(self, input_dim, gat_hidden, lstm_hidden, output_dim=1, num_heads=2):
        super(LSTM_GAT, self).__init__()
        self.gat = GATConv(input_dim, gat_hidden, heads=num_heads, concat=True)
        self.lstm_input_dim = gat_hidden * num_heads
        self.lstm = nn.LSTM(input_size=self.lstm_input_dim, hidden_size=lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, output_dim)

    def forward(self, x_seq, edge_index):
        """
        x_seq: [batch_size, num_nodes, seq_len, input_dim]
        edge_index: 图的连接 (2, num_edges)
        """
        batch_size, num_nodes, seq_len, input_dim = x_seq.size()

        gat_inputs = x_seq[:, :, -1, :]  # 只用最近一天的特征做图构建 [B, N, F]
        gat_inputs = gat_inputs.reshape(-1, input_dim)       # [B*N, F]

        edge_index_batch = self.expand_edge_index(edge_index, batch_size, num_nodes)

        # Apply GAT to last-day node features
        gat_out = self.gat(gat_inputs, edge_index_batch)  # [B*N, D]
        gat_out = gat_out.view(batch_size, num_nodes, -1)  # [B, N, D]

        # Expand GAT features along sequence length
        lstm_input = gat_out.unsqueeze(2).repeat(1, 1, seq_len, 1)  # [B, N, T, D]
        lstm_input = lstm_input.view(batch_size * num_nodes, seq_len, -1)

        lstm_out, _ = self.lstm(lstm_input)  # [B*N, T, H]
        final_hidden = lstm_out[:, -1, :]  # [B*N, H]

        output = self.fc(final_hidden)  # [B*N, 1]
        return output.view(batch_size, num_nodes)

    def expand_edge_index(self, edge_index, batch_size, num_nodes):
        """将 edge_index 复制到每个 batch，并做 index 偏移"""
        edge_index_list = []
        for i in range(batch_size):
            offset = i * num_nodes
            edge_index_list.append(edge_index + offset)
        return torch.cat(edge_index_list, dim=1)

# 使用范例
# # 假设 5 支股票，输入序列为 20 天，每天只有收盘价（1维特征）
# batch_size = 16
# num_nodes = 5
# seq_len = 20
# input_dim = 1

# x_seq = torch.randn(batch_size, num_nodes, seq_len, input_dim)

# # 定义股票图（例如按行业或相似性连接）edge_index为（2，num_edges）
# edge_index = torch.tensor([
#     [0, 1, 1, 2, 3, 4],
#     [1, 0, 2, 1, 4, 3]
# ], dtype=torch.long)

# model = LSTM_GAT(
#     input_dim=input_dim,
#     gat_hidden=8,
#     lstm_hidden=16,
#     output_dim=1,
#     num_heads=2
# )

# out = model(x_seq, edge_index)
# print(out.shape)  # [batch_size, num_nodes] → 每支股票预测第21天价格


class GRUCellOriginal(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCellOriginal, self).__init__()
        self.hidden_size = hidden_size

        # 门控权重
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size)

        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size)

        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t, h_prev):
        z_t = torch.sigmoid(self.W_z(x_t) + self.U_z(h_prev))
        r_t = torch.sigmoid(self.W_r(x_t) + self.U_r(h_prev))
        h_tilde = torch.tanh(self.W_h(x_t) + self.U_h(r_t * h_prev))
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCellOriginal(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):  # x: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(x.size(1)):
            x_t = x[:, t, :]
            h_t = self.cell(x_t, h_t)

        output = self.fc(h_t)  # 输出第21天预测值
        return output

# model = GRU(input_size=1, hidden_size=64)

# x = torch.randn(100, 20, 1)   # 100支股票，20天股价序列
# y = torch.randn(100, 1)       # 第21天真实价格

# pred = model(x)
# loss = F.mse_loss(pred, y)
# loss.backward()


class GRU_GAT(nn.Module):
    def __init__(self, input_size, gru_hidden, gat_hidden, output_size=1, num_heads=2):
        super(GRU_GAT, self).__init__()
        self.gru_hidden = gru_hidden
        self.gru_cell = GRUCellOriginal(input_size, gru_hidden)
        self.gat = GATConv(gru_hidden, gat_hidden, heads=num_heads, concat=True)
        self.regressor = nn.Linear(gat_hidden * num_heads, output_size)

    def forward(self, x_seq, edge_index):
        """
        x_seq: (num_nodes, seq_len, input_size)
        edge_index: (2, num_edges)
        """
        num_nodes, seq_len, _ = x_seq.shape
        h = torch.zeros(num_nodes, self.gru_hidden, device=x_seq.device)

        # GRU 时间序列建模
        for t in range(seq_len):
            x_t = x_seq[:, t, :]  # 每一时刻的所有节点输入
            h = self.gru_cell(x_t, h)

        # GAT 空间建图处理（图神经网络）
        h_gat = self.gat(h, edge_index)

        # 输出层回归预测
        out = self.regressor(h_gat)  # (num_nodes, 1)
        return out.squeeze(-1)

# # 假设你有100支股票，每支股票过去20天股价序列（1维）
# x_seq = torch.randn(100, 20, 1)  # (num_nodes, seq_len, input_size)

# # 构造图结构 edge_index，例如基于行业或相关性
# edge_index = torch.tensor([
#     [0, 1, 2, 3, 3, 4],
#     [1, 0, 3, 2, 4, 3]
# ], dtype=torch.long)

# model = GRU_GAT(input_size=1, gru_hidden=64, gat_hidden=32, num_heads=2)
# pred = model(x_seq, edge_index)  # 输出形状 (num_nodes,)



class RNNCellWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super(RNNCellWithDropout, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.W_x = nn.Linear(input_size, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x_t, h_prev):
        x_t_dropped = self.dropout(x_t)  # 只对输入使用 Dropout
        h_t = torch.tanh(self.W_x(x_t_dropped) + self.W_h(h_prev))
        return h_t


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, dropout=0.5):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = RNNCellWithDropout(input_size, hidden_size, dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):  # x: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(x.size(1)):
            x_t = x[:, t, :]
            h_t = self.rnn_cell(x_t, h_t)

        out = self.fc(h_t)  # 用最终隐藏状态回归
        return out

# model = RNN(input_size=1, hidden_size=64, dropout=0.5)

# x = torch.randn(100, 20, 1)  # 100支股票，20天股价历史
# y = torch.randn(100, 1)      # 第21天目标价格

# pred = model(x)
# loss = F.mse_loss(pred, y)
# loss.backward()
