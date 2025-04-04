import torch.nn as nn
from easydict import EasyDict

def build_act(model_cfg:EasyDict):
    """
    根据 model_cfg.activation 返回对应的激活层实例。
    
    参数:
        model_cfg (dict): 包含模型配置的字典，其中 'activation' 键指定所需的激活函数名称。

    返回:
        nn.Module: 对应的激活层实例。
    """
    activation_name = model_cfg.get('activation', 'Tanh')  # 默认使用Tanh
    
    activation_map = {
        'ReLU': nn.ReLU,
        'Tanh': nn.Tanh,
        'Hardtanh': nn.Hardtanh,
        'Sigmoid': nn.Sigmoid,
        'LeakyReLU': nn.LeakyReLU,
        'ELU': nn.ELU,
        'GELU': nn.GELU,
        'Softmax': lambda: nn.Softmax(dim=1),  # 指定dim以避免错误
        'Softplus': nn.Softplus,
        'Hardshrink': nn.Hardshrink,
        'Hardsigmoid': nn.Hardsigmoid,
        'HardSwish': nn.Hardswish
    }
    
    if activation_name not in activation_map:
        raise ValueError(f"Unsupported activation: {activation_name}. "
                         f"Supported activations are: {list(activation_map.keys())}")
    
    return activation_map[activation_name]()
