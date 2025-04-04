import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr  
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from scipy.stats import spearmanr
import numpy as np

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
def train_model(model, optimizer, train_data, val_data, epochs, lr, patience=10):
    print("Training...")
   
    # 创建学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 创建一个列表来存储每个epoch的损失值
    loss_values = []
    val_loss_values = []
    IC = []
    
    # 早停相关变量
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ic = []
        
        for i in train_data:
            x, edge_index, edge_attr, y, fadj, sadj = i[0], i[1], i[2], i[3], i[4], i[5]
            y_pred = model(x=x, edge_index=edge_index, sadj=sadj, fadj=fadj)
            optimizer.zero_grad()
            y = y.view(-1,1)

            mse_loss = F.mse_loss(y_pred, y)
            loss = mse_loss
            ic, rank_ic, icir, rankicir = calculate_metrics(y_pred, y)
            epoch_loss += loss.item()

            if not np.isnan(ic):
                epoch_ic.append(ic)

            loss.backward()
            optimizer.step()

        # 平均损失
        avg_loss = epoch_loss / len(train_data)
        loss_values.append(avg_loss)
        
        # 验证集评估
        val_loss = evaluate_model(model, val_data)
        val_loss_values.append(val_loss)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        # 如果连续patience次验证损失没有改善，则早停
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            model.load_state_dict(best_model_state)
            break
        
        # 计算每个epoch的IC平均值
        if epoch_ic:
            avg_ic = np.nanmean(epoch_ic)
        else:
            avg_ic = np.nan
        IC.append(avg_ic)
        
        # 输出训练进度
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {val_loss:.4f}, IC: {avg_ic:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
    return loss_values, val_loss_values, IC

# 模型验证
def evaluate_model(model, val_data):
    model.eval()  # 切换到评估模式
    val_loss = 0.0
    with torch.no_grad():  # 在验证时不需要梯度计算
        for i in val_data:
            x, edge_index, edge_attr, y, fadj, sadj = i[0], i[1], i[2], i[3], i[4], i[5]
            y_pred = model(x=x, edge_index=edge_index, sadj=sadj, fadj=fadj)
            
            y = y.view(-1, 1)
            
            # 模型预测
            # n = emb1.size(0)
            mse_loss = F.mse_loss(y_pred, y)
            # loss_dep = (loss_dependence(emb1, com1, n) + loss_dependence(emb2, com2, n))/2
            # loss_com = common_loss(com1,com2)
            
            # # 计算验证集损失
            # loss = mse_loss + beta * loss_dep + theta * loss_com
            loss = mse_loss
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
            x, edge_index, edge_attr, y, fadj, sadj = i[0], i[1], i[2], i[3], i[4], i[5]
            y_pred = model(x=x, edge_index=edge_index, sadj=sadj, fadj=fadj)
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
