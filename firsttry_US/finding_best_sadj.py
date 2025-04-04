import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import *
from data.generate_data import create_data, normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr  
from torch.optim.lr_scheduler import ReduceLROnPlateau
from main import stock_data, calculate_metrics, train_model, test_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设定数据量
n_company = 110
num_days = 1002  # 股价天数
num_features = 6  # 每个时间点有6个特征（如开盘价、收盘价等）
time_window = 18  #时间窗口设置为18天为最优

# 使用最优超参数
beta = 1.2164610467572498e-08
theta = 1.3510247718823894e-10
epochs = 38
batch_size = n_company
learning_rate = 5.0049754318915105e-05
hidden_dim1 = 125
hidden_dim2 = 260
hidden_dim3 = 193
hidden_dim4 = 248
dropout = 0.8993095644168758
# 获取结构邻接矩阵
correlation_path = "/root/firsttry-main/firsttry_US/data/topology_matrix.csv"
correlation_matrix = pd.read_csv(correlation_path)

# 定义阈值调优函数
def tune_threshold(correlation_matrix, thresholds, stock_data, time_window, beta, theta, epochs, learning_rate):
    results = []  # 用于存储每个阈值的结果
    
    for threshold in thresholds:
        print(f"Testing threshold: {threshold}")
        
        # 处理邻接矩阵
        sadj = correlation_matrix.iloc[0:n_company, 1:(n_company+1)].values
        sadj[abs(sadj) < threshold] = 0  # 将相关系数小于阈值的边删除
        sadj = torch.tensor(sadj, dtype=torch.float32) 
        sadj = sadj + sadj.T.mul(sadj.T > sadj) - sadj.mul(sadj.T > sadj)
        sadj = normalize(sadj + torch.eye(sadj.size(0), dtype=torch.float32))
        sadj = torch.from_numpy(sadj)
        
        # 创建训练数据集和测试集
        batch_data_list = create_data(stock_data=stock_data, sadj=sadj, time_window=time_window)
        train_data, temp_data = train_test_split(batch_data_list, test_size=0.4, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        # 创建模型
        model = SFGCN(input_dim=time_window, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, 
                      hidden_dim3=hidden_dim3, hidden_dim4=hidden_dim4, dropout=dropout).to(device)
        
        # 训练模型
        _, val_loss_values, _ = train_model(model, train_data, val_data, epochs=epochs, lr=learning_rate, beta=beta, theta=theta)
        
        # 测试模型
        pred_values, target_values, mse_loss, mae_loss, rmse_loss, mape_loss = test_model(model, test_data)
        pred_values, target_values = [t.cpu() for t in pred_values], [t.cpu() for t in target_values]
        
        # 计算评价指标
        ic, rank_ic, icir, rankicir = calculate_metrics(pred_values, target_values)
        
        # 记录结果
        results.append({
            'threshold': threshold,
            'mse_loss': mse_loss,
            'mae_loss': mae_loss,
            'rmse_loss': rmse_loss,
            'mape_loss': mape_loss,
            'ic': ic,
            'rank_ic': rank_ic,
            'icir': icir,
            'rankicir': rankicir
        })
        
        # 打印当前阈值的结果
        print(f"Threshold: {threshold}, "
              f"MSE Loss: {mse_loss:.6f}, "
              f"MAE Loss: {mae_loss:.6f}, "
              f"RMSE Loss: {rmse_loss:.6f}, "
              f"MAPE Loss: {mape_loss:.6f}, "
              f"IC: {ic:.6f}, "
              f"Rank IC: {rank_ic:.6f}, "
              f"ICIR: {icir:.6f}, "
              f"Rank ICIR: {rankicir:.6f}")
    
    # 将结果保存到DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存到CSV文件
    results_df.to_csv('threshold_tuning_results.csv', index=False)
    print("Results saved to 'threshold_tuning_results.csv'")
    
    return results_df

# 定义阈值范围
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 调优阈值
results_df = tune_threshold(correlation_matrix, thresholds, stock_data, time_window, beta, theta, epochs, learning_rate)

# 打印所有结果
print(results_df)