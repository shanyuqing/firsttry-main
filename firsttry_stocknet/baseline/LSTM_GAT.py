import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
import sys
import os
import matplotlib.pyplot as plt
import numpy as np 
import torch.nn.functional as F
from model_config import Lstm_gat_Config 
from scipy.stats import spearmanr 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import train_data, test_data, val_data
from baseline.utils_models import train_model, test_model,calculate_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM_GAT_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, num_heads=4):
        super(LSTM_GAT_Model, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        
        # GAT layer
        self.gat1 = GATConv(hidden_size, hidden_size, heads=num_heads, dropout=0.5)
        self.gat2 = GATConv(hidden_size * num_heads, output_size, heads=1, dropout=0.5)
        
    def forward(self, x, edge_index):
        # x: Node features (batch_size, seq_len, num_nodes, feature_dim)
        # edge_index: Graph connectivity (edge_index[0], edge_index[1])
        
        # First, process through LSTM (Time-series part)
        x, _ = self.lstm(x)  
        
        # GAT layer
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        
        return x
if __name__ == "__main__":
    # Hyperparameters
    input_size = Lstm_gat_Config.input_size   
    hidden_size = Lstm_gat_Config.hidden_size 
    output_size = Lstm_gat_Config.output_size  
    num_heads = Lstm_gat_Config.num_heads
    lr = Lstm_gat_Config.lr
    num_nodes = Lstm_gat_Config.num_nodes   
    epochs = Lstm_gat_Config.epochs
    num_layers = Lstm_gat_Config.num_layers 


    model  = LSTM_GAT_Model(input_size, hidden_size, output_size, num_layers, num_heads).to(device)

    # 训练模型
    loss_values, val_loss_values, IC = train_model(model, train_data, val_data, epochs=epochs, lr=lr)

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
    plt.savefig('/root/firsttry-main/firsttry_stocknet/baseline/LSTM+GAT.png')

    # 测试模型
    pred_values, target_values = test_model(model, test_data)
    pred_values, target_values = [t.cpu() for t in pred_values], [t.cpu() for t in target_values]

    # 计算评价指标
    ic, rank_ic, icir, rankicir = calculate_metrics(pred_values, target_values)

    # 输出四个评价指标
    print(f'lstm+gat_IC: {ic}')
    print(f'lstm+gat_RankIC: {rank_ic}')
    print(f'lstm+gat_ICIR: {icir}')
    print(f'lstm+gat_RankICIR: {rankicir}')

    # 绘制实际值与预测值对比
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
    plt.savefig('/root/firsttry-main/firsttry_stocknet/baseline/LSTM+GAT_testing_real_forecast.png')

# test mse_loss: 0.04570084078663803
# test mae_loss: 0.20656050059450679
# test rmse_loss: 0.21280892438677898
# test mape_loss: 41.081169189698244
# lstm+gat_IC: 2.691460364818894e-05
# lstm+gat_RankIC: 0.005036186700666941
# lstm+gat_ICIR: 0.0004880087224379663
# lstm+gat_RankICIR: 6.125283468338102e-07