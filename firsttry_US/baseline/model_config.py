from easydict import EasyDict

# 最优值

# 定义 Base_Config
Base_Config = EasyDict({
    'input_size': 18,
    "output_size" : 1,# Output for each node (or aggregated output)
})
  
Gat_Config = EasyDict({
    "lr": 0.007060465495557617,
    "hidden_size": 97,
    "dropout": 0.6335058322104097,
    "epochs": 86,
    "num_heads": 6,
    'Base': True  
})

Gru_Config = EasyDict({
    "lr": 0.004221795019072234,
    "hidden_size": 208,
    "dropout": 0.7054222168853704,
    "epochs": 251,
    "num_layers": 3,
    'Base': True  
})

Gru_gat_Config = EasyDict({
    "lr": 0.004221795019072234,
    "hidden_size": 208,
    "dropout": 0.7054222168853704,
    "epochs": 251,
    "num_heads": 3,
    "num_layers": 3,
    'Base': True  
})

Lstm_Config = EasyDict({
    "lr": 0.005567212291777722,
    "hidden_size": 208,
    "dropout": 0.44087028910199877,
    "epochs": 63,
    "num_heads": 2,
    "num_layers": 1,
    'Base': True
})

Lstm_gat_Config = EasyDict({
    "lr": 0.0010568781452759718,
    "hidden_size": 463,
    "dropout": 0.16712312612463165,
    "epochs": 87,
    "num_heads": 5,
    "num_layers": 1,
    'Base': True  
})

Rnn_Config = EasyDict({
    "lr": 0.0004808791853689083,
    "hidden_size": 323,
    "dropout": 0.3425711515596825,
    "epochs": 194,
    "num_heads": 1,
    "num_layers": 5,
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

