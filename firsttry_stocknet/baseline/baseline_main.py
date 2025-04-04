import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
import sys
import os
import matplotlib.pyplot as plt
import numpy as np 
import torch.nn.functional as F
from scipy.stats import spearmanr  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import train_data, test_data, val_data
from model_config import Gat_Config, Gru_Config, Gru_gat_Config, Lstm_Config, Lstm_gat_Config, Rnn_Config
from baseline.utils_models import train_model, test_model,calculate_metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = [Gat_Config, Gru_Config, Gru_gat_Config, Lstm_Config, Lstm_gat_Config, Rnn_Config]

for i in config:
    # Hyperparameters
    input_size = i.input_size 
    hidden_size = Gat_Config.hidden_size
    output_size = Gat_Config.output_size
    num_heads = Gat_Config.num_heads
    lr = Gat_Config.lr
    num_nodes = Gat_Config.num_nodes 
    epochs = Gat_Config.epochs
    dropout = Gat_Config.dropout

    model = GATModel(input_size, hidden_size, output_size, num_heads, dropout).to(device)

    # 训练模型
    loss_values, val_loss_values, IC = train_model(model, train_data, val_data, epochs, lr)

    # 绘制训练时的损失曲线
    epochs_range = range(epochs)
    plt.plot(epochs_range, loss_values, marker='o', linestyle='-', color='b', label='train_loss')
    plt.plot(epochs_range, val_loss_values, marker='o', linestyle='-', color='g', label='val_loss')
    plt.plot(epochs_range, IC, marker='o', linestyle='-', color='r', label='ic')
    plt.title('Loss/IC Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss/IC', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig('/root/firsttry-main/firsttry_stocknet/baseline/GAT.png')

    # 测试模型
    pred_values, target_values = test_model(model, test_data)
    pred_values, target_values = [t.cpu() for t in pred_values], [t.cpu() for t in target_values]

    # 计算评价指标
    ic, rank_ic, icir, rankicir = calculate_metrics(pred_values, target_values)

    # 输出四个评价指标
    print(f'gat_IC: {ic}')
    print(f'gat_RankIC: {rank_ic}')
    print(f'gat_ICIR: {icir}')
    print(f'gat_RankICIR: {rankicir}')

    plt.figure()
    plt.style.use('ggplot')
    # 创建折线图
    plt.plot(target_values, label='real', color='blue')  # 实际值
    plt.plot(pred_values, label='forecast', color='red', linestyle='--')  # 预测值

    # 增强视觉效果
    plt.grid(True)
    plt.title('real vs forecast')
    plt.ylabel('value')
    plt.legend()
    plt.savefig('/root/firsttry-main/firsttry_stocknet/baseline/GRU_testing_real_forecast.png')

