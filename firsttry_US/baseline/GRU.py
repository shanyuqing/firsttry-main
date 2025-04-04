import torch
import torch.nn as nn
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import train_data, test_data, val_data
from model_config import Gru_Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils_models import train_model, test_model,calculate_metrics

# GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        out, _ = self.gru(x)
        # Take the output of the last time step
        out = self.fc(out)  
        return out
if __name__ == "__main__":
    # Hyperparameters
    input_size = Gru_Config.input_size 
    hidden_size = Gru_Config.hidden_size
    output_size = Gru_Config.output_size  
    num_layers = Gru_Config.num_layers
    lr = Gru_Config.lr
    num_nodes = Gru_Config.num_nodes  
    epochs = Gru_Config.epochs


    model = GRUModel(input_size, hidden_size, output_size, num_layers).to(device)

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
    plt.savefig('/root/firsttry-main/firsttry_US/baseline/GRU.png')

    # 测试模型
    pred_values, target_values = test_model(model, test_data)
    pred_values, target_values = [t.cpu() for t in pred_values], [t.cpu() for t in target_values]

    # 计算评价指标
    ic, rank_ic, icir, rankicir = calculate_metrics(pred_values, target_values)

    # 输出四个评价指标
    print(f'gru_IC: {ic}')
    print(f'gru_RankIC: {rank_ic}')
    print(f'gru_ICIR: {icir}')
    print(f'gru_RankICIR: {rankicir}')

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
    plt.savefig('/root/firsttry-main/firsttry_US/baseline/GRU_testing_real_forecast.png')

# test mse_loss: 0.0009029853638724854
# test mae_loss: 0.020760833654495967
# test rmse_loss: 0.02709325685681123
# test mape_loss: 4.468054278852976
# gru_IC: 0.030914373823922564
# gru_RankIC: 0.025310940839863506
# gru_ICIR: 1.069085228310731
# gru_RankICIR: 2.8598774709882813e-06