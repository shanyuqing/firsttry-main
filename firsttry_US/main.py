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
from utils_models import train_model, test_model, evaluate_model, calculate_all_metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设定数据量
n_company = 110
num_days = 1002  # 股价天数
num_features = 6  # 每个时间点有6个特征（如开盘价、收盘价等）
time_window = 18  #时间窗口设置为18天为最优

# 使用最优超参数
beta = 1.2164610467572498e-08
theta = 1.3510247718823894e-10
epochs = 60
# 38
batch_size = n_company
lr = 5.0049754318915105e-05
hidden_dim1 = 125
hidden_dim2 = 260
hidden_dim3 = 193
hidden_dim4 = 248
dropout = 0.8993095644168758

## 获取股票数据 
folder_path = '/root/firsttry-main/firsttry_US/CMIN/preprocessed_US'

## 获取文件夹中所有文件的文件名
file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

## 初始化一个空的DataFrame用于存储合并后的数据
all_data = pd.DataFrame()

##迭代每个文件，读取文件并添加文件名列
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path, sep='\t', header=None, names=['dt', 'open', 'high', 'low', 'close', 'adj close', 'volume'])  # 读取文件
    # 按行倒序
    df = df.iloc[::-1].reset_index(drop=True)
    df['code'] = file_name[0:-4]  # 在DataFrame中添加一列，用于存储文件名
    all_data = pd.concat([all_data, df], ignore_index=True)  # 合并当前文件数据到总数据

# 初始化数组存储股价数据
stock_data = np.zeros((n_company, num_days, num_features))          # 数值特征数组

# 填充数据
for i, stock_code in enumerate(all_data['code'].unique()):
    stock_df = all_data[all_data['code'] == stock_code]
    # 存储数值特征
    stock_data[i] = stock_df.iloc[:, 1:-1].values  # -1排除code列，1:-1取数值列

# # 数据归一化(整体进行归一)
# normalized_data = np.zeros_like(stock_data)

# # 遍历所有特征列（跳过第一列时间戳，处理第2到第7列）
# for features_index in range(stock_data.shape[2]):  # 从1开始，跳过时间戳列
#     # 提取所有公司和所有天数的当前特征数据
#     stock_type_data = stock_data[:, :, features_index]
    
#     # 计算全局最大值和最小值
#     global_min = np.min(stock_type_data)
#     global_max = np.max(stock_type_data)
    
#     # 对该特征进行 Min-Max 归一化
#     normalized_data[:, :, features_index] = (stock_type_data - global_min) / (global_max - global_min)
# stock_data = normalized_data


# 获取结构邻接矩阵
correlation_path="/root/firsttry-main/firsttry_US/data/topology_matrix.csv"
correlation_matrix = pd.read_csv(correlation_path)
sadj = correlation_matrix.iloc[0:n_company, 1:(n_company+1)].values
sadj[abs(sadj) < 0.7] = 0  # 阈值取0.7为最佳
sadj = torch.tensor(sadj, dtype=torch.float32) 
sadj = sadj + sadj.T.mul(sadj.T > sadj) - sadj.mul(sadj.T > sadj)
sadj = normalize(sadj + torch.eye(sadj.size(0), dtype=torch.float32))
sadj = torch.from_numpy(sadj)

# 创建训练数据集和测试集
batch_data_list = create_data(stock_data=stock_data, sadj=sadj, time_window= time_window)
train_data, temp_data = train_test_split(batch_data_list, test_size=0.4, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# 计算 IC, RankIC, ICIR, RankICIR
def calculate_metrics(list1, list2):
   
    values1 = [t.item() for t in list1]
    values2 = [t.item() for t in list2]
    
    # 如果预测值和真实值完全相同，直接返回 IC 和 RankIC 为 1（完美相关）
    if np.allclose(values1, values2):
        return 1.0, 1.0, 1.0, 1.0
    
    # 计算IC
    ic = np.corrcoef(values1, values2)[0, 1] if np.std(values1) != 0 else np.nan # 使用 NumPy 的 corrcoef 函数
   
    # 如果相关系数不可用（例如，返回 NaN），则返回 0
    if np.isnan(ic):
        ic = 0
    
    # 计算RankIC
    rank_ic, _ = spearmanr(values1, values2)  # 使用 SciPy 的 spearmanr 函数
    if np.isnan(rank_ic):
        rank_ic = 0

    # 计算 ICIR 和 RankICIR
    ic_std = np.std(np.array(values1) - np.array(values2))  # IC 标准差
    rank_ic_std = np.std(np.argsort(values1) - np.argsort(values2))  # RankIC 标准差

    icir = ic / ic_std if ic_std != 0 else np.nan  # 防止除以零
    rankicir = rank_ic / rank_ic_std if rank_ic_std != 0 else np.nan  # 防止除以零

    return ic, rank_ic, icir, rankicir

def calculate_ic(pred, target):
    """计算信息系数 (Information Coefficient)"""
    pred = pred.view(-1).detach().cpu().numpy()
    target = target.view(-1).detach().cpu().numpy()
    return np.corrcoef(pred, target)[0, 1]  # Pearson相关系数

# 定义训练模型
# def train_model(model, train_data, val_data, epochs, lr):
#     """模型训练函数
    
#     Args:
#         model: 待训练模型
#         train_data: 训练集 DataLoader
#         val_data: 验证集 DataLoader
#         epochs: 训练轮次
#         lr: 初始学习率
        
#     Returns:
#         tuple: (训练损失记录, 验证指标记录, IC指标记录)
#     """
#     # 初始化优化器和学习率调度器
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
#     # 训练记录初始化
#     history = {
#         'train_loss': [],
#         'train_ic': [],
#         'val_metrics': []
#     }

#     # 训练循环
#     for epoch in range(epochs):
#         model.train()
#         epoch_metrics = {
#             'total_loss': 0.0,
#             'total_ic': 0.0,
#             'num_samples': 0
#         }
        
#         # 训练批次迭代
#         for batch in train_data:
#             x, edge_index, edge_attr, y, fadj, sadj = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
#             optimizer.zero_grad()
            
#             # 模型前向传播
#             y_pred = model(x, sadj, fadj, edge_index)
#             y = y.view(-1, 1)
            
#             # 损失计算
#             mse_loss = F.mse_loss(y_pred, y)
#             loss = mse_loss 
            
#             # 反向传播
#             loss.backward()
#             optimizer.step()
            
#             # 指标记录
#             batch_size = y.size(0)
#             epoch_metrics['total_loss'] += loss.item() * batch_size
#             epoch_metrics['num_samples'] += batch_size
            
#             # IC指标计算
#             with torch.no_grad():
#                 batch_ic = calculate_ic(y_pred, y)  # 建议封装成独立函数
#                 epoch_metrics['total_ic'] += batch_ic * batch_size

#         # 计算epoch平均指标
#         avg_loss = epoch_metrics['total_loss'] / epoch_metrics['num_samples']
#         avg_ic = epoch_metrics['total_ic'] / epoch_metrics['num_samples']
#         history['train_loss'].append(avg_loss)
#         history['train_ic'].append(avg_ic)
        
#         # 验证集评估
#         val_metrics = evaluate_model(model, val_data)  # 使用优化后的评估函数
#         history['val_metrics'].append(val_metrics)
        
#         # 学习率调度
#         scheduler.step(val_metrics['val_mse'])  # 根据验证集MSE调整学习率
        
#         # 训练过程打印
#         print(f"Epoch {epoch+1}/{epochs} | "
#               f"Train Loss: {avg_loss:.4f} | "
#               f"Train IC: {avg_ic:.4f} | "
#               f"Val MSE: {val_metrics['val_mse']:.4f} | "
#               f"Val MAE: {val_metrics['val_mae']:.4f} | "
#               f"Val MAPE: {val_metrics['val_mape']:.2f}%")
    
#     return history['train_loss'], history['val_metrics'], history['train_ic']
# # 模型验证
# def evaluate_model(model, val_data):
#     """评估模型在验证集上的性能，返回多指标结果
    
#     Args:
#         model: 待评估模型
#         val_data: 验证集 DataLoader
    
#     Returns:
#         dict: 包含各指标的平均值
#     """
#     model.eval()
#     total_losses = {
#         'mse': 0.0,
#         'mae': 0.0,
#         'mape': 0.0,
#         'num_samples': 0
#     }
#     all_preds = []
#     all_targets = []
    
#     with torch.no_grad():
#         for batch in val_data:
#             x, edge_index, edge_attr, y, fadj, sadj = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
#             y_pred = model(x, sadj, fadj, edge_index)
#             y = y.view(-1, 1)
            
#             # 保存预测和真实值用于整体计算
#             all_preds.append(y_pred)
#             all_targets.append(y)
            
#             # 逐batch累加指标
#             total_losses['mse'] += F.mse_loss(y_pred, y, reduction='sum').item()
#             total_losses['mae'] += F.l1_loss(y_pred, y, reduction='sum').item()
            
#             # 处理 MAPE 的除零问题
#             non_zero_mask = y != 0
#             if non_zero_mask.any():
#                 mape_batch = torch.abs((y[non_zero_mask] - y_pred[non_zero_mask]) / y[non_zero_mask]).sum().item()
#                 total_losses['mape'] += mape_batch * 100  # 转换为百分比
                
#             total_losses['num_samples'] += y.size(0)
    
#     # 合并所有样本统一计算指标
#     all_preds = torch.cat(all_preds, dim=0)
#     all_targets = torch.cat(all_targets, dim=0)
#     n_samples = total_losses['num_samples']
    
#     # 最终指标计算
#     mse = total_losses['mse'] / n_samples
#     mae = total_losses['mae'] / n_samples
#     rmse = torch.sqrt(torch.tensor(mse)).item()
#     mape = total_losses['mape'] / n_samples if total_losses['mape'] > 0 else float('nan')
    
#     return {
#         'val_mse': mse,
#         'val_mae': mae,
#         'val_rmse': rmse,
#         'val_mape': mape
#     }


# # 模型测试
# def test_model(model, test_data):
#     """测试模型性能，返回预测值和多指标结果
    
#     Args:
#         model: 训练好的模型
#         test_data: 测试集 DataLoader
    
#     Returns:
#         tuple: (pred_values, target_values, metrics_dict)
#     """
#     model.eval()
#     total_losses = {
#         'mse': 0.0,
#         'mae': 0.0,
#         'mape': 0.0,
#         'num_samples': 0
#     }
#     pred_values = []
#     target_values = []
    
#     with torch.no_grad():
#         for batch in test_data:
#             x, edge_index, edge_attr, y, fadj, sadj = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
#             y_pred = model(x, sadj, fadj, edge_index)
#             y = y.view(-1, 1)
            
#             # 保存预测和真实值
#             pred_values.append(y_pred)
#             target_values.append(y)
            
#             # 累加指标
#             total_losses['mse'] += F.mse_loss(y_pred, y, reduction='sum').item()
#             total_losses['mae'] += F.l1_loss(y_pred, y, reduction='sum').item()
            
#             # 处理 MAPE
#             non_zero_mask = y != 0
#             if non_zero_mask.any():
#                 mape_batch = torch.abs((y[non_zero_mask] - y_pred[non_zero_mask]) / y[non_zero_mask]).sum().item()
#                 total_losses['mape'] += mape_batch * 100
                
#             total_losses['num_samples'] += y.size(0)
    
#     # 合并数据
#     pred_values = torch.cat(pred_values, dim=0)
#     target_values = torch.cat(target_values, dim=0)
#     n_samples = total_losses['num_samples']
    
#     # 计算最终指标
#     mse = total_losses['mse'] / n_samples
#     mae = total_losses['mae'] / n_samples
#     rmse = torch.sqrt(torch.tensor(mse)).item()
#     mape = total_losses['mape'] / n_samples if total_losses['mape'] > 0 else float('nan')
    
#     metrics = {
#         'test_mse': mse,
#         'test_mae': mae,
#         'test_rmse': rmse,
#         'test_mape': mape
#     }
    
#     return pred_values, target_values, metrics

if __name__ == "__main__":
    print("main.py is being run directly")
    # 创建模型
    model = SFGCN(input_dim = time_window, hidden_dim1 = hidden_dim1, hidden_dim2 = hidden_dim2, hidden_dim3=hidden_dim3, hidden_dim4=hidden_dim4, dropout = dropout).to(device)

    # # 训练模型
    # loss_values, val_loss_values, IC = train_model(model, train_data, val_data, epochs, lr)

    # # 绘制训练时的损失曲线
    # epochs_range = range(epochs)
    # val_mse_values = [d['val_mse'] for d in val_loss_values]
    # plt.figure(figsize=(8, 6))
    # plt.plot(epochs_range, loss_values, marker='o', linestyle='-', color='b', label='train_loss')
    # plt.plot(epochs_range,val_mse_values, marker='o', linestyle='-', color='g', label='val_loss')
    # plt.plot(epochs_range, IC, marker='o', linestyle='-', color='r', label='ic')
    # plt.title('Loss/IC Curve', fontsize=14)
    # plt.xlabel('Epoch', fontsize=12)
    # plt.ylabel('Loss/IC', fontsize=12)
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    # plt.savefig('main.png')

    # # 验证阶段
    # val_metrics = evaluate_model(model, val_data)
    # print(f"Validation MSE: {val_metrics['val_mse']:.4f}, MAPE: {val_metrics['val_mape']:.2f}%")

    # # 测试阶段
    # pred_values, target_values, test_metrics = test_model(model, test_data)
    # pred_values, target_values = [t.cpu() for t in pred_values], [t.cpu() for t in target_values]
    
    # # 计算评价指标
    # ic, rank_ic, icir, rankicir = calculate_metrics(pred_values, target_values)
    
    # print(f"Test RMSE: {test_metrics['test_rmse']:.4f}, MAE: {test_metrics['test_mae']:.4f}, RMSE:{test_metrics['test_rmse']:.4f},MAPE:{test_metrics['test_mape']:.2f}%")
    # print(f"IC:{ic:.4f},RANK_IC:{rank_ic:.4f} ICIR:{icir:.4f} RANKICIR:{rankicir:.4f}")
    
    
    # 训练模型
    loss_values, val_loss_values, IC = train_model(model, train_data, val_data, epochs, lr)

    # # 绘制训练时的损失曲线
    # epochs_range = range(epochs)
    # val_mse_values = [d['val_mse'] for d in val_loss_values]
    # plt.figure(figsize=(8, 6))
    # plt.plot(epochs_range, loss_values, marker='o', linestyle='-', color='b', label='train_loss')
    # plt.plot(epochs_range,val_mse_values, marker='o', linestyle='-', color='g', label='val_loss')
    # plt.plot(epochs_range, IC, marker='o', linestyle='-', color='r', label='ic')
    # plt.title('Loss/IC Curve', fontsize=14)
    # plt.xlabel('Epoch', fontsize=12)
    # plt.ylabel('Loss/IC', fontsize=12)
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    # plt.savefig('main.png')

    # 验证阶段
    val_metrics = evaluate_model(model, val_data)
    print(f"Validation MSE: {val_metrics['val_mse']:.4f}, MAPE: {val_metrics['val_mape']:.2f}%")

    # 测试阶段
    pred_values, target_values, test_metrics = test_model(model, test_data)
    pred_values, target_values = [t.cpu() for t in pred_values], [t.cpu() for t in target_values]
    
    # 计算评价指标
    ic, rank_ic, icir, rankicir = calculate_all_metrics(pred_values, target_values)
    
    print(f"Test RMSE: {test_metrics['test_rmse']:.4f}, MAE: {test_metrics['test_mae']:.4f}, RMSE:{test_metrics['test_rmse']:.4f},MAPE:{test_metrics['test_mape']:.2f}%")
    print(f"IC:{ic:.4f},RANK_IC:{rank_ic:.4f} ICIR:{icir:.4f} RANKICIR:{rankicir:.4f}")
    
    





