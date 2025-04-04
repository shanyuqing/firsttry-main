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
from utils_models import calculate_all_metrics, train_model, evaluate_model, test_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads, dropout):
        super(GATModel, self).__init__()
        
        # 第一层GAT：多注意力头+拼接
        self.gat1 = GATConv(
            in_channels=input_size,
            out_channels=hidden_size,
            heads=num_heads,         # 使用num_heads个注意力头
            dropout=dropout,
            concat=True               # 默认True，拼接多头结果
        )
        
        # 层间Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 第二层GAT：单注意力头（输出层）+平均
        self.gat2 = GATConv(
            in_channels=hidden_size * num_heads,  # 输入维度是第一层的输出
            out_channels=output_size,
            heads=1,                # 输出层使用单头
            dropout=dropout,
            concat=False             # 不拼接（对多头结果取平均）
        )

    def forward(self, x, edge_index):
        # 第一层
        x = self.gat1(x, edge_index)
        x = F.elu(x)                 # 更常用的GAT激活函数
        x = self.dropout(x)
        
        # 第二层
        x = self.gat2(x, edge_index)
        return x  
    
if __name__ == "__main__":
    # Hyperparameters
    input_size = Gat_Config.input_size 
    hidden_size = Gat_Config.hidden_size
    output_size = Gat_Config.output_size
    num_heads = Gat_Config.num_heads
    lr = Gat_Config.lr
    epochs = Gat_Config.epochs
    dropout = Gat_Config.dropout

    model = GATModel(input_size, hidden_size, output_size, num_heads, dropout).to(device)

    # 训练模型
    loss_values, val_loss_values, IC = train_model(model, train_data, val_data, epochs, lr)

    # 绘制训练时的损失曲线
    epochs_range = range(epochs)
    val_mse_values = [d['val_mse'] for d in val_loss_values]
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, loss_values, marker='o', linestyle='-', color='b', label='train_loss')
    plt.plot(epochs_range,val_mse_values, marker='o', linestyle='-', color='g', label='val_loss')
    plt.plot(epochs_range, IC, marker='o', linestyle='-', color='r', label='ic')
    plt.title('Loss/IC Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss/IC', fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig('gat.png')

    # 验证阶段
    val_metrics = evaluate_model(model, val_data)
    print(f"Validation MSE: {val_metrics['val_mse']:.4f}, MAPE: {val_metrics['val_mape']:.2f}%")

    # 测试阶段
    pred_values, target_values, test_metrics = test_model(model, test_data)
    pred_values, target_values = [t.cpu() for t in pred_values], [t.cpu() for t in target_values]
    
    # 计算评价指标
    ic, rank_ic, icir, rankicir = calculate_all_metrics(pred_values, target_values)
    
    print(f"Test RMSE: {test_metrics['test_rmse']:.4f}, MAE: {test_metrics['test_mae']:.4f}, RMSE:{test_metrics['test_rmse']:.4f},MAPE:{test_metrics['test_mape']:.2f}%")
    print(f"IC:{ic:.4f},RANK_IC:{rank_ic:.4f} ICIR:{icir:.4f} RANK_ICIR:{rankicir:.4f}")
    
    


