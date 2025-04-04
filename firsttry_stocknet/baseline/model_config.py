from easydict import EasyDict
from baseline_models import GATModel, GRU_GAT_Model, GRUModel, LSTM_GAT_Model, LSTMModel, RNNModel

# 定义 Base_Config
Base_Config = EasyDict({
    'input_size': 15,
    "output_size" : 1,# Output for each node (or aggregated output)
    "num_nodes" : 81  # Number of nodes (example size)
})

# todo
# configs = [Gat_Config,...]
# model = config.model_cls(*params)
# config.model_name

# 定义其他 config
Gat_Config = EasyDict({
    "model_name": "GAT",
    "model_cls": GATModel,
    "lr": 0.004151171103747412,
    "hidden_size": 292,
    "dropout": 0.7134496354145927,
    "epochs": 140,
    "num_heads": 6,
    'Base': True  
})

# GRU
Gru_Config = EasyDict({
    "model_name": "GRU",
    "model_cls": GRUModel,
    "lr": 0.004434001139740843,
    "hidden_size": 114,
    "dropout": 0.6571000954514259,
    "epochs": 133,
    "num_layers": 3,
    'Base': True  
})

Gru_gat_Config = EasyDict({
    "model_name": "GRU_GAT",
    "model_cls": GRU_GAT_Model,
    "lr": 0.0006954533247944113,
    "hidden_size": 204,
    "dropout": 0.28001123528965594,
    "epochs": 223,
    "num_heads": 5,
    "num_layers": 3,
    'Base': True  
})
# LSTM
Lstm_Config = EasyDict({
    "model_name": "LSTM",
    "model_cls": LSTMModel,
    "model":"lstm",
    "lr": 0.009510788689886657,
    "hidden_size": 498,
    "dropout": 0.15675712618550203,
    "epochs": 139,
    "num_layers": 5,
    'Base': True
})

Lstm_gat_Config = EasyDict({
    "model_name": "LSTM_GAT",
    "model_cls": LSTM_GAT_Model,
    "lr": 0.00642792358050644,
    "hidden_size": 488,
    "dropout": 0.1661570378810997,
    "epochs": 145,
    "num_heads": 2,
    "num_layers": 5,
    'Base': True  
})

Rnn_Config = EasyDict({
    "model_name": "RNN",
    "model_cls": RNNModel,
    "lr": 0.005414096104373523,
    "hidden_size": 220,
    "dropout": 0.8119377974379395,
    "epochs": 43,
    "num_layers": 4,
    'Base': True  
})
# 更新函数
def update_config_with_base(config, base_config):
    """
    如果 config 中 'Base' 键为 True，使用 base_config 更新 config。
    
    参数：
    - config: 待更新的 EasyDict 配置对象
    - base_config: Base 配置对象，用于更新
    
    返回：
    - 更新后的 config
    """
    if config.get('Base', False):  # 如果 'Base' 为 True
        config.update(base_config)  # 使用 base_config 更新 config
    return config

# 使用函数更新 config
Gat_Config = update_config_with_base(Gat_Config, Base_Config)
Gru_Config = update_config_with_base(Gru_Config, Base_Config)
Gru_gat_Config = update_config_with_base(Gru_gat_Config, Base_Config)
Lstm_Config = update_config_with_base(Lstm_Config, Base_Config)
Lstm_gat_Config = update_config_with_base(Lstm_gat_Config, Base_Config)
Rnn_Config = update_config_with_base(Rnn_Config, Base_Config)


if __name__ == '__main__':
    # 输出更新后的 config
    print(Gat_Config)
    print(Gru_Config)
    print(Gru_gat_Config)
    print(Lstm_Config)
    print(Lstm_gat_Config)
    print(Rnn_Config)

