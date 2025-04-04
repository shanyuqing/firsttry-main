# 基准实验超参数优化
import os
import sys
import torch
import torch.optim as optim
import optuna
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import train_data, val_data,test_data
from utils_models import train_model, evaluate_model, test_model, calculate_metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from GRU import GRUModel

input_size = 18
output_size = 1
def objective(trial):
    # 定义超参数搜索空间
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    hidden_size = trial.suggest_int("hidden_size", 64, 512)
    dropout = trial.suggest_float("dropout", 0.1, 0.9)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    epochs = trial.suggest_int("epochs", 10, 256)  # 添加 epochs 超参数
    num_heads = trial.suggest_int("num_heads", 1, 6)
    num_layers = trial.suggest_int("num_layers", 1, 6)
    # 创建模型

    model = GRUModel(input_size, hidden_size, output_size, num_layers).to(device)

    # 定义优化器
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-3)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5e-3)

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
print("gru Best trial:")
trial = study.best_trial
print(f"  Value (Validation Loss): {trial.value:.4f}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# gru Best trial:
#   Value (Validation Loss): 0.0010
#   Params: 
#     learning_rate: 0.004221795019072234
#     hidden_size: 208
#     dropout: 0.7054222168853704
#     optimizer: SGD
#     epochs: 251
#     num_heads: 3
#     num_layers: 3