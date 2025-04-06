from .models import GAT, GRU, GRU_GAT, LSTM, LSTM_GAT, RNN
from torch.nn import Module
from .model_ablation import ST_GraphAttn, ST_GraphAttn_ablation_1, ST_GraphAttn_ablation_2, ST_GraphAttn_ablation_3, ST_GraphAttn_ablation_4, ST_GraphAttn_ablation_5

__all__ = {
    'ST_GraphAttn': ST_GraphAttn,
    'GAT': GAT,
    'GRU': GRU,
    'GRU_GAT': GRU_GAT,
    "LSTM": LSTM,
    "LSTM_GAT": LSTM_GAT,
    "RNN": RNN,
    "ST_GraphAttn_ablation_1": ST_GraphAttn_ablation_1,
    "ST_GraphAttn_ablation_2": ST_GraphAttn_ablation_2,
    "ST_GraphAttn_ablation_3": ST_GraphAttn_ablation_3,
    "ST_GraphAttn_ablation_4": ST_GraphAttn_ablation_4,
    "ST_GraphAttn_ablation_5": ST_GraphAttn_ablation_5
}


def build_network(model_cfg):
    model:Module = __all__[model_cfg.NAME](model_cfg)
    return model