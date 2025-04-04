# 基准实验超参数优化
import os
import sys
import torch
import torch.optim as optim
import optuna
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import train_data, val_data,test_data
from baseline.utils_models import train_model, evaluate_model, test_model, calculate_metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from RNN import RNNModel

input_size = 11
output_size = 1

def objective(trial):
    # 定义超参数搜索空间
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_int("hidden_size", 32, 300)
    dropout = trial.suggest_float("dropout", 0.1, 0.9)
    epochs = trial.suggest_int("epochs", 10, 256)  # 添加 epochs 超参数
    num_heads = trial.suggest_int("num_heads", 1, 6)
    num_layers = trial.suggest_int("num_layers", 1, 6)
    # 创建模型

    model = RNNModel(input_size, hidden_size, output_size, num_layers).to(device)

    # 训练模型
    loss_values, val_loss_values, IC = train_model(
        model, train_data, val_data, epochs=epochs, lr=learning_rate)

    # 返回验证集损失（Optuna 会最小化该值）
    return min(val_loss_values)

# 创建 Optuna 学习
study = optuna.create_study(direction="minimize")

# 运行调优
study.optimize(objective, n_trials=50)

# 输出最佳超参数
print("Best trial:")
trial = study.best_trial
print(f"  Value (Validation Loss): {trial.value:.4f}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Best trial:
#   Value (Validation Loss): 0.0008
#   Params: 
#     learning_rate: 0.008497973607726963
#     hidden_size: 39
#     dropout: 0.7977800118471927
#     epochs: 98
#     num_heads: 6
#     num_layers: 1