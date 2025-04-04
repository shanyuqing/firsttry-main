from .models import GAT, GRU, GRU_GAT, LSTM, LSTM_GAT, RNN
from torch.nn import Module
from .models_2 import ST_GraphAttn

__all__ = {
    'ST_GraphAttn': ST_GraphAttn,
    'GAT': GAT,
    'GRU': GRU,
    'GRU_GAT': GRU_GAT,
    "LSTM": LSTM,
    "LSTM_GAT": LSTM_GAT,
    "RNN": RNN
}


def build_network(model_cfg):
    model:Module = __all__[model_cfg.NAME](model_cfg)
    return model