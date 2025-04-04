from easydict import EasyDict

# 定义 Base_Config
Base_Config = EasyDict({
    'input_size': 12,
    "output_size" : 1,# Output for each node (or aggregated output)
    "num_nodes" : 253  # Number of nodes (example size)
})

# 定义其他 config，例如 config_gat
Gat_Config = EasyDict({
    "lr": 0.008186951945594845,
    "hidden_size": 180,
    "dropout": 0.21751498769856137,
    "epochs": 180,
    "num_heads": 5,
    'Base': True  
})

Gru_Config = EasyDict({
    "lr": 0.008400310804597983,
    "hidden_size": 125,
    "dropout": 0.718473230685683,
    "epochs": 47,
    "num_layers": 1,
    'Base': True  
})

Gru_gat_Config = EasyDict({
    "lr": 0.00847137946813196,
    "hidden_size": 107,
    "dropout": 0.5358674820751936,
    "epochs": 37,
    "num_heads": 2,
    "num_layers": 2,
    'Base': True  
})

Lstm_Config = EasyDict({
    "lr": 0.009879224880679395,
    "hidden_size": 105,
    "dropout": 0.5684131803112665,
    "epochs": 149,
    "num_layers": 1,
    'Base': True
})

Lstm_gat_Config = EasyDict({
    "lr": 0.0001764336382667243,
    "hidden_size": 229,
    "dropout": 0.5046007010131169,
    "epochs": 174,
    "num_heads": 5,
    "num_layers": 1,
    'Base': True  
})

Rnn_Config = EasyDict({
    "lr": 0.008497973607726963,
    "hidden_size": 39,
    "dropout": 0.7977800118471927,
    "epochs": 98,
    "num_layers": 1,
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

