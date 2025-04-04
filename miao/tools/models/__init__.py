from .models import MarketMixer, MLP_miner, GRU_miner, SimpleFeatsMixer
from torch.nn import Module

__all__ = {
    'MarketMixer': MarketMixer,
    'MLP_miner': MLP_miner,
    'GRU_miner': GRU_miner,
    'SimpleFeatsMixer': SimpleFeatsMixer
}


def build_network(model_cfg, dataset):
    model:Module = __all__[model_cfg.NAME](model_cfg, dataset)
    return model