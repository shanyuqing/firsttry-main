from torch.nn import Linear, Module, Dropout, GRU, Tanh, Hardswish
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from torch import nn
import torch

from .utils import build_act

# model definition
class MLP_miner(Module):
    # define model elements
    def __init__(self, model_cfg, subset:torch.utils.data.Subset):
        super().__init__()
        N_stocks = subset.dataset.X.shape[1]

        if "HIDDEN_DIM" in model_cfg.keys():
            hidden_dim = model_cfg.HIDDEN_DIM
        else:
            hidden_dim = 16
        print("hidden_dim: ", hidden_dim)
        self.hidden = Linear(N_stocks, hidden_dim, bias=False)
        self.dropout = Dropout(p=0.5)
        self.output = Linear(hidden_dim, N_stocks)
        self.act1 = build_act(model_cfg)
        kaiming_uniform_(self.hidden.weight, mode='fan_in', nonlinearity='tanh')
        
    def forward(self, X):
        X = self.act1(self.hidden(X))
        X = self.dropout(X)
        X = self.output(X)
        return X


class GRU_miner(Module):
    '''用GRU提取特征的尝试，由于速度太慢暂时不用'''
    def __init__(self, n_inputs, n_outputs, hidden_dim=64, num_layers=1):
        super(GRU_miner, self).__init__()
        self.gru_encoder = GRU(
            input_size=n_inputs,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.5
        )
        self.mlp_miner = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = Dropout(p=0.5)
        self.act1 = Tanh()
        self.output_layer = Linear(hidden_dim, n_outputs)
        kaiming_uniform_(self.mlp_miner.weight, mode='fan_in', nonlinearity='tanh')

    def forward(self, inputs):
        _, support = self.gru_encoder(inputs)  # (num_layers, hidden_dim)
        support = support.squeeze()
        h_alphas = self.mlp_miner(support)  
        h_alphas = self.act1(h_alphas)
        h_alphas = self.dropout(h_alphas)
        logits = self.output_layer(h_alphas)
        return logits
    

class NoGraphMixer(nn.Module):
    def __init__(self, n_stocks, hidden_dim, emdedding_dim=16):
        '''
        hidden_dim: 本层的隐藏层维度
        emdedding_dim: 之前对于每个stock的 emdedding 维度
        '''
        self.hidden_dim = hidden_dim
        super(NoGraphMixer, self).__init__()
        self.dense1 = nn.Linear(n_stocks, hidden_dim)
        self.activation = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, n_stocks)
        self.layer_norm_stock = nn.LayerNorm(n_stocks)
        self.layer_norm_hidden = nn.LayerNorm(emdedding_dim)
        kaiming_uniform_(self.dense1.weight, mode='fan_in', nonlinearity='tanh')
        kaiming_uniform_(self.dense2.weight, mode='fan_in', nonlinearity='tanh')

    def forward(self, inputs):
        x = inputs
        x = x.permute(0, 2, 1)
        x = self.layer_norm_stock(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = x.permute(0, 2, 1)
        x = self.layer_norm_hidden(x)
        return x


class MarketMixer(nn.Module):
    '''简化了StockMixer模型结构，去除时序Mix，使之可以并行训练'''
    def __init__(self, model_cfg, subset:torch.utils.data.Subset):
        super(MarketMixer, self).__init__()
        N_stocks, N_feats = subset.dataset.X.shape[1], subset.dataset.X.shape[2]

        if "HIDDEN_DIM" in model_cfg.keys():
            hidden_dim = model_cfg.HIDDEN_DIM
        else:
            hidden_dim = 16

        if "MARKET_HIDDEN_DIM" in model_cfg.keys():
            market_hidden_dim = model_cfg.MARKET_HIDDEN_DIM
        else:
            market_hidden_dim = 16

        if "ALPHA" in model_cfg.keys():
            self.alpha = model_cfg.ALPHA
        else:
            self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # 对feats信息进行提取
        self.feats_mixer = nn.Linear(N_feats, hidden_dim)
        # 在 stock 维度进行 Dense
        self.stock_mixer = NoGraphMixer(N_stocks, market_hidden_dim, emdedding_dim=hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        self.act = build_act(model_cfg)
        # self.dropout = nn.Dropout(p=0.5)
        kaiming_uniform_(self.feats_mixer.weight, mode='fan_in', nonlinearity='tanh')
        kaiming_uniform_(self.output_layer.weight, mode='fan_in', nonlinearity='tanh')

    
    def forward(self, inputs):
        feats_h = self.feats_mixer(inputs)  # x_n.shape: (batch, N, hidden_dim)
        feats_h = self.act(feats_h)
        # h = self.dropout(h)
        # cs_h = self.stock_mixer(feats_h)
        # combined_h = self.alpha*feats_h + self.alpha*cs_h
        out = self.output_layer(feats_h)
        return out


class SimpleFeatsMixer(nn.Module):
    '''对于Featrues聚合到1维，然后再用MLP处理'''
    def __init__(self, model_cfg, subset:torch.utils.data.Subset):
        super().__init__()
        N_stocks, N_feats = subset.dataset.X.shape[1], subset.dataset.X.shape[2]

        if "HIDDEN_DIM" in model_cfg.keys():
            hidden_dim = model_cfg.HIDDEN_DIM
        else:
            hidden_dim = 16
        feats_embed_dim = 8
        self.feats_mixer = nn.Linear(N_feats, feats_embed_dim)
        self.hidden_layer = nn.Linear(N_stocks*feats_embed_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, N_stocks)

        self.dropout = Dropout(p=0.5)
        self.act = build_act(model_cfg)
        
    def forward(self, inputs):
        feats_h = self.feats_mixer(inputs).squeeze()  # x_n.shape: (batch, N, hidden_dim)
        feats_h = feats_h.flatten(1)  # (batch, N_stock * feats_embed_dim)
        feats_h = self.hidden_layer(feats_h)
        feats_h = self.act(feats_h)
        feats_h = self.dropout(feats_h)
        out = self.output_layer(feats_h)
        return out