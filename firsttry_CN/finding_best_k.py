import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import *
from data.generate_data_for_k import create_data, normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr  
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设定数据量
n_company = 253
num_days = 970  # 股价天数
num_features = 6  # 每个时间点有6个特征（如开盘价、收盘价等）
time_window = 11  #时间窗口设置为11天为最优

# 使用最优超参数
beta = 2.8544226393364566e-08
theta = 4.613700739288068e-10
epochs = 29
batch_size = n_company
learning_rate = 1.1104666123517542e-05
hidden_dim1 = 97
hidden_dim2 = 134
hidden_dim3 = 190
hidden_dim4 = 65
dropout = 0.8483006964002686

## 获取股票数据 
folder_path = '/root/firsttry-main/firsttry_CN/preprocessed_deleted'

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

# 数据归一化(整体进行归一)
normalized_data = np.zeros_like(stock_data)

# 遍历所有特征列（跳过第一列时间戳，处理第2到第7列）
for features_index in range(stock_data.shape[2]):  # 从1开始，跳过时间戳列
    # 提取所有公司和所有天数的当前特征数据
    stock_type_data = stock_data[:, :, features_index]
    
    # 计算全局最大值和最小值
    global_min = np.min(stock_type_data)
    global_max = np.max(stock_type_data)
    
    # 对该特征进行 Min-Max 归一化
    normalized_data[:, :, features_index] = (stock_type_data - global_min) / (global_max - global_min)
stock_data = normalized_data


# 获取结构邻接矩阵
correlation_path="/root/firsttry-main/firsttry_CN/data/topology_matrix_CN(cleaned)_qwen.csv"
correlation_matrix = pd.read_csv(correlation_path)
sadj = correlation_matrix.iloc[0:n_company, 1:(n_company+1)].values
sadj[abs(sadj) < 0.3] = 0  # 
sadj = torch.tensor(sadj, dtype=torch.float32) 
sadj = sadj + sadj.T.mul(sadj.T > sadj) - sadj.mul(sadj.T > sadj)
sadj = normalize(sadj + torch.eye(sadj.size(0), dtype=torch.float32))
sadj = torch.from_numpy(sadj)

# 定义损失函数
def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2)**2)
    return cost

def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC

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

# 定义训练模型
def train_model(model, train_data, val_data, epochs, lr, beta, theta):
    print("Training...")
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=5e-3)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
   
    # 创建一个列表来存储每个epoch的损失值
    loss_values = []
    val_loss_values = []  # 存储验证集的损失
    IC = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ic = []  # 每个epoch记录的IC值列表
        for i in train_data:
            x, edge_index, edge_attr, y, fadj, sadj= i[0], i[1], i[2], i[3], i[4], i[5]
            y_pred, emb1, com1, com2, emb2, emb= model(x, sadj, fadj, edge_index)
            optimizer.zero_grad()
            y=y.view(-1,1)

            # 模型预测
            n = emb1.size(0)
            mse_loss = F.mse_loss(y_pred, y)
            # mae_loss = F.l1_loss(y_pred, y)
            # rmse_loss = torch.sqrt(mse_loss)
            # mape_loss = torch.mean(torch.abs((y - y_pred) / (y + 1e-8))) * 100
            loss_dep = (loss_dependence(emb1, com1, n) + loss_dependence(emb2, com2, n))/2
            loss_com = common_loss(com1,com2)
            loss = mse_loss + beta * loss_dep + theta * loss_com
            ic, rank_ic, icir, rankicir = calculate_metrics(y_pred, y)
            epoch_loss += loss.item()

            if not np.isnan(ic):
                epoch_ic.append(ic)  # 记录每个batch的IC

            loss.backward()
            optimizer.step()

        # 平均损失
        avg_loss = epoch_loss / len(train_data)
        loss_values.append(avg_loss)
        # 验证集评估
        val_loss = evaluate_model(model, val_data, beta, theta)
        val_loss_values.append(val_loss)
        
        # 计算每个epoch的IC平均值
        if epoch_ic:
            avg_ic = np.nanmean(epoch_ic)
        else:
            avg_ic = np.nan
        IC.append(avg_ic)  # 将每个epoch的IC值添加到IC列表

        scheduler.step(val_loss)  # 传入验证损失进行学习率调整

        # # 输出每个epoch的训练损失和验证损失
        # print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f},IC: {avg_ic:.4f}')
        
    return loss_values, val_loss_values, IC

# 模型验证
def evaluate_model(model, val_data, beta, theta):
    model.eval()  # 切换到评估模式
    val_loss = 0.0
    with torch.no_grad():  # 在验证时不需要梯度计算
        for i in val_data:
            x, edge_index, edge_attr, y, fadj, sadj = i[0], i[1], i[2], i[3], i[4], i[5]
            y_pred, emb1, com1, com2, emb2, emb = model(x, sadj, fadj, edge_index)
            
            y = y.view(-1, 1)
            
            # 模型预测
            n = emb1.size(0)
            mse_loss = F.mse_loss(y_pred, y)
            # mae_loss = F.l1_loss(y_pred, y)
            # rmse_loss = torch.sqrt(mse_loss)
            # mape_loss = torch.mean(torch.abs((y - y_pred) / (y + 1e-8))) * 100
            loss_dep = (loss_dependence(emb1, com1, n) + loss_dependence(emb2, com2, n))/2
            loss_com = common_loss(com1,com2)
            
            # 计算验证集损失
            loss = mse_loss + beta * loss_dep + theta * loss_com
            val_loss += loss.item()
    
    # 返回验证集平均损失
    avg_val_loss = val_loss / len(val_data)
    return avg_val_loss


# 模型测试
def test_model(model, test_data):
    print("Testing...")
    model.eval()
    mse_total_loss = 0.0
    mae_total_loss = 0.0
    rmse_total_loss = 0.0
    mape_total_loss = 0.0
    pred_values = []
    target_values = []
    with torch.no_grad():
        for i in test_data:
            x, edge_index, edge_attr, y, fadj, sadj= i[0], i[1], i[2], i[3], i[4], i[5]
            y_pred, emb1, com1, com2, emb2, emb= model(x, sadj, fadj, edge_index)
            y = y.view(-1, 1)
            mse_loss = F.mse_loss(y_pred, y)
            mae_loss = F.l1_loss(y_pred, y)
            rmse_loss = torch.sqrt(mse_loss)
            non_zero_mask = y != 0  # 过滤掉真实值为零的样本
            mape_loss = torch.mean(torch.abs((y[non_zero_mask] - y_pred[non_zero_mask]) / (y[non_zero_mask] + 1e-8))) * 100
            mse_total_loss += mse_loss.item()
            mae_total_loss += mae_loss.item()
            rmse_total_loss += rmse_loss.item()
            mape_total_loss += mape_loss.item()
            pred_values.append(y_pred)
            target_values.append(y)
    # 拼接所有预测值和目标值
    pred_values = torch.cat(pred_values, dim=0)
    target_values = torch.cat(target_values, dim=0)
    mse_loss=mse_total_loss / len(test_data)
    mae_loss=mae_total_loss / len(test_data)
    rmse_loss=rmse_total_loss / len(test_data)
    mape_loss=mape_total_loss / len(test_data)
    # print(f"main_test mse_loss: {mse_total_loss / len(test_data)}")
    # print(f"main_test mae_loss: {mae_total_loss / len(test_data)}")
    # print(f"main_test rmse_loss: {rmse_total_loss / len(test_data)}")
    # print(f"main_test mape_loss: {mape_total_loss / len(test_data)}")
    return pred_values, target_values,mse_loss,mae_loss,rmse_loss, mape_loss

def tune_k_parameter(stock_data, sadj, time_window, k_values):
    results = []
    
    for k in k_values:
        print(f"Testing k={k}")
        
        # 生成数据（传入当前 k 值）
        batch_data_list = create_data(stock_data=stock_data, sadj=sadj, time_window=time_window, k=k)
        
        # 划分数据集
        train_data, temp_data = train_test_split(batch_data_list, test_size=0.4, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        # 创建模型（使用你的模型定义）
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
            'k': k,
            'mse_loss': mse_loss,
            'mae_loss': mae_loss,
            'rmse_loss': rmse_loss,
            'mape_loss': mape_loss,
            'ic': ic,
            'rank_ic': rank_ic,
            'icir': icir,
            'rankicir': rankicir
        })
        
        # 打印结果
        print(f"k={k}, "
              f"MSE: {mse_loss:.6f}, MAE: {mae_loss:.6f}, RMSE: {rmse_loss:.6f}, "
              f"MAPE: {mape_loss:.6f}, IC: {ic:.6f}, RankIC: {rank_ic:.6f}, "
              f"ICIR: {icir:.6f}, RankICIR: {rankicir:.6f}")
    
    # 保存结果到 CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('k_tuning_results.csv', index=False)
    print("Results saved to 'k_tuning_results.csv'")
    
    return results_df

# 定义 k 的调优范围（例如 1 到 10）
k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# 运行调优
results_df = tune_k_parameter(stock_data, sadj, time_window, k_values)

# 打印最优结果
best_k_row = results_df.loc[results_df['mse_loss'].idxmin()]
print(f"Best k: {best_k_row['k']}, "
      f"Best MSE: {best_k_row['mse_loss']:.6f}")