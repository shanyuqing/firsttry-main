import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
import sys
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np 
from scipy.stats import spearmanr 
from model_config import Gru_gat_Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import train_data, test_data,  val_data
from utils_models import calculate_all_metrics, train_model, evaluate_model, test_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRU_GAT_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, dropout):
        super(GRU_GAT_Model, self).__init__()
        
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # GAT layer
        self.gat1 = GATConv(hidden_size, hidden_size, num_heads, dropout)
        self.gat2 = GATConv(hidden_size * num_heads, output_size, heads=1, dropout)
        
    def forward(self, x, edge_index):
        # x: Node features (batch_size, seq_len, num_nodes, feature_dim)
        # edge_index: Graph connectivity (edge_index[0], edge_index[1])
        
        # First, process through GRU (Time-series part)
        x, _ = self.gru(x)  # x: (batch_size, seq_len, hidden_size)
       
        # GAT layer
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        
        return x

if __name__ == "__main__":
    # Hyperparameters
    input_size = Gru_gat_Config.input_size  
    hidden_size = Gru_gat_Config.hidden_size
    output_size = Gru_gat_Config.output_size 
    num_heads = Gru_gat_Config.num_heads
    lr = Gru_gat_Config.lr
    epochs = Gru_gat_Config.epochs
    num_layers=Gru_gat_Config.num_layers
    dropout = Gru_gat_Config.dropout

    model = GRU_GAT_Model(input_size, hidden_size, output_size, num_layers, num_heads,dropout).to(device)

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
    plt.savefig('/root/firsttry-main/firsttry_US/baseline/GRU+GAT.png')

    # 测试模型
    pred_values, target_values = test_model(model, test_data)
    pred_values, target_values = [t.cpu() for t in pred_values], [t.cpu() for t in target_values]

    # 计算评价指标
    ic, rank_ic, icir, rankicir = calculate_metrics(pred_values, target_values)

    # 输出四个评价指标
    print(f'gru+gat_IC: {ic}')
    print(f'gru+gat_RankIC: {rank_ic}')
    print(f'gru+gat_ICIR: {icir}')
    print(f'gru+gat_RankICIR: {rankicir}')

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
    plt.savefig('/root/firsttry-main/firsttry_US/baseline/GRU+GAT_testing_real_forecast.png')

# test mse_loss: 0.10246615718842157
# test mae_loss: 0.3187651905008984
# test rmse_loss: 0.31969393797332263
# test mape_loss: 66.66168574996406
# gru+gat_IC: -0.00471483741485748
# gru+gat_RankIC: -0.002643462012315142
# gru+gat_ICIR: -0.1600206727983759
# gru+gat_RankICIR: -3.0118624787605394e-07