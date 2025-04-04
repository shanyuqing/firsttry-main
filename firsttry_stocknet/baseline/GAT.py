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
from model_config import Gat_Config
from baseline.utils_models import train_model, test_model,calculate_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GATModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads,dropout):
        super(GATModel, self).__init__()
        
        self.gat1 = GATConv(input_size, hidden_size, num_heads, dropout)
        self.gat2 = GATConv(hidden_size * num_heads, output_size, num_heads, dropout)
        
    def forward(self, x, edge_index):
        # First GAT layer
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        
        # Second GAT layer
        x = self.gat2(x, edge_index)
        
        return x

if __name__ == "__main__":
    # Hyperparameters
    input_size = Gat_Config.input_size 
    hidden_size = Gat_Config.hidden_size
    output_size = Gat_Config.output_size
    num_heads = Gat_Config.num_heads
    lr = Gat_Config.lr
    num_nodes = Gat_Config.num_nodes 
    epochs = Gat_Config.epochs
    dropout = Gat_Config.dropout

    model = GATModel(input_size, hidden_size, output_size, num_heads, dropout).to(device)

    # 训练模型
    loss_values, val_loss_values = train_model(model, train_data, val_data, epochs, lr)

    # # 绘制训练时的损失曲线
    # epochs_range = range(epochs)
    # plt.plot(epochs_range, loss_values, marker='o', linestyle='-', color='b', label='train_loss')
    # plt.plot(epochs_range, val_loss_values, marker='o', linestyle='-', color='g', label='val_loss')
    # plt.plot(epochs_range, IC, marker='o', linestyle='-', color='r', label='ic')
    # plt.title('Loss/IC Curve', fontsize=14)
    # plt.xlabel('Epoch', fontsize=12)
    # plt.ylabel('Loss/IC', fontsize=12)
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    # plt.savefig('/root/firsttry-main/firsttry_stocknet/baseline/GAT.png')

    # 测试模型
    pred_values, target_values = test_model(model, test_data)
    pred_values, target_values = [t.cpu() for t in pred_values], [t.cpu() for t in target_values]

    # 计算评价指标
    # ic, rank_ic, icir, rankicir = calculate_metrics(pred_values, target_values)

    # # 输出四个评价指标
    # print(f'gat_IC: {ic}')
    # print(f'gat_RankIC: {rank_ic}')
    # print(f'gat_ICIR: {icir}')
    # print(f'gat_RankICIR: {rankicir}')

    # plt.figure()
    # plt.style.use('ggplot')
    # # 创建折线图
    # plt.plot(target_values, label='real', color='blue')  # 实际值
    # plt.plot(pred_values, label='forecast', color='red', linestyle='--')  # 预测值

    # # 增强视觉效果
    # plt.grid(True)
    # plt.title('real vs forecast')
    # plt.ylabel('value')
    # plt.legend()
    # plt.savefig('/root/firsttry-main/firsttry_stocknet/baseline/GRU_testing_real_forecast.png')

# Testing...
# test mse_loss: 0.010739396574035526
# test mae_loss: 0.09516852989373915
# test rmse_loss: 0.09933379662024927
# test mape_loss: 18.917958711524566
# gat_IC: -0.002891149303868599
# gat_RankIC: -0.004714272088920687
# gat_ICIR: -0.060369398563758614
# gat_RankICIR: -5.711466465884298e-07