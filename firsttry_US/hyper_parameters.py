# 学习率（learning_rate）：1e-5 到 1e-2（对数尺度）。
# 隐藏层维度（hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4）：64 到 512。
# 丢弃率（dropout）：0.1 到 0.9。
# 正则化参数（beta, theta）：1e-10 到 1e-3（对数尺度）。
# 优化器类型：Adam 或 SGD。
# epochs 5到50
from model import *
import torch.optim as optim
import optuna
from main import train_data, val_data,test_data, train_model, test_model, calculate_metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time_window=15

def objective(trial):
    # 定义超参数搜索空间
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    hidden_dim1 = trial.suggest_int("hidden_dim1", 64, 512)
    hidden_dim2 = trial.suggest_int("hidden_dim2", 64, 512)
    hidden_dim3 = trial.suggest_int("hidden_dim3", 64, 512)
    hidden_dim4 = trial.suggest_int("hidden_dim4", 64, 512)
    dropout = trial.suggest_float("dropout", 0.1, 0.9)
    beta = trial.suggest_float("beta", 1e-10, 1e-3, log=True)
    theta = trial.suggest_float("theta", 1e-10, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    epochs = trial.suggest_int("epochs", 5, 50)  # 添加 epochs 超参数

    # 创建模型
    model = SFGCN(
        input_dim=time_window,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        hidden_dim3=hidden_dim3,
        hidden_dim4=hidden_dim4,
        dropout=dropout
    ).to(device)

    # 定义优化器
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-3)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5e-3)

    # 训练模型
    loss_values, val_loss_values, IC = train_model(
        model, train_data, val_data, epochs=epochs, lr=learning_rate, beta=beta, theta=theta
    )

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

# 使用最佳超参数重新训练模型
best_params = study.best_params
model = SFGCN(
    input_dim=time_window,
    hidden_dim1=best_params["hidden_dim1"],
    hidden_dim2=best_params["hidden_dim2"],
    hidden_dim3=best_params["hidden_dim3"],
    hidden_dim4=best_params["hidden_dim4"],
    dropout=best_params["dropout"]
).to(device)

# 训练模型
loss_values, val_loss_values, IC = train_model(
    model, train_data, val_data, epochs=best_params["epochs"], 
    lr=best_params["learning_rate"], beta=best_params["beta"], 
    theta=best_params["theta"]
)

# 测试模型
pred_values, target_values, mse_loss, mae_loss, rmse_loss, mape_loss = test_model(model, test_data)

# 计算评价指标
ic, rank_ic, icir, rankicir = calculate_metrics(pred_values, target_values)

print(f"MSE Loss: {mse_loss:.4f}")
print(f"MAE Loss: {mae_loss:.4f}")
print(f"RMSE Loss: {rmse_loss:.4f}")
print(f"MAPE Loss: {mape_loss:.4f}")
print(f"IC: {ic:.4f}")
print(f"Rank IC: {rank_ic:.4f}")
print(f"ICIR: {icir:.4f}")
print(f"Rank ICIR: {rankicir:.4f}")


# 绘制优化历史
optuna.visualization.plot_optimization_history(study)

# 绘制超参数重要性
optuna.visualization.plot_param_importances(study)


# learning_rate: 5.0049754318915105e-05
#     hidden_dim1: 125
#     hidden_dim2: 260
#     hidden_dim3: 193
#     hidden_dim4: 248
#     dropout: 0.8993095644168758
#     beta: 1.2164610467572498e-08
#     theta: 1.3510247718823894e-10
#     optimizer: Adam
#     epochs: 38
