import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import matplotlib.pyplot as plt 
import numpy as np 
import torch.nn.functional as F
from model_config import Rnn_Config
from scipy.stats import spearmanr  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import train_data, test_data, val_data
from baseline.utils_models import train_model, test_model, calculate_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        out, _ = self.rnn(x)
        # Take the output of the last time step
        out = self.fc(out) 
        return out
if __name__ == "__main__":
    # Hyperparameters
    input_size = Rnn_Config.input_size  
    hidden_size = Rnn_Config.hidden_size
    output_size = Rnn_Config.output_size  
    num_layers = Rnn_Config.num_layers
    lr = Rnn_Config.lr
    num_nodes = Rnn_Config.num_nodes  
    epochs = Rnn_Config.epochs

    model = RNNModel(input_size, hidden_size, output_size, num_layers).to(device)

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
    plt.savefig('/root/firsttry-main/firsttry_stocknet/baseline/RNN.png')

    # 测试模型
    pred_values, target_values = test_model(model, test_data)
    pred_values, target_values = [t.cpu() for t in pred_values], [t.cpu() for t in target_values]

    # 计算评价指标
    ic, rank_ic, icir, rankicir = calculate_metrics(pred_values, target_values)

    # 输出四个评价指标
    print(f'rnn_IC: {ic}')
    print(f'rnn_RankIC: {rank_ic}')
    print(f'rnn_ICIR: {icir}')
    print(f'rnn_RankICIR: {rankicir}')

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
    plt.savefig('/root/firsttry-main/firsttry_stocknet/baseline/RNN_testing_real_forecast.png')

# Testing...
# test mse_loss: 0.15944438957306276
# test mae_loss: 0.37534914090930216
# test rmse_loss: 0.3992584431506544
# test mape_loss: 74.6766024669969
# rnn_IC: 0.003843211837845607
# rnn_RankIC: 0.006861226603531761
# rnn_ICIR: 0.028210584017282467
# rnn_RankICIR: 8.323587908532663e-07