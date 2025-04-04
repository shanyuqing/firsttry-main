import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import spearmanr
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

def calculate_all_metrics(preds, targets):
    """计算所有相关指标"""
    # IC计算
    ic = safe_corrcoef(preds, targets)
    
    # Rank IC计算
    rank_ic = safe_spearmanr(preds, targets)
    
    # ICIR计算（需要时间序列数据）
    # 假设每个样本代表一个时间点
    ic_series = []
    for i in range(1, len(preds)):
        ic_series.append(np.corrcoef(preds[:i], targets[:i])[0,1])
    icir = np.nanmean(ic_series) / (np.nanstd(ic_series) + 1e-8) if len(ic_series) > 0 else 0
    
    # Rank ICIR计算
    rank_ic_series = []
    for i in range(1, len(preds)):
        rank_ic_series.append(spearmanr(preds[:i], targets[:i]).correlation)
    rank_icir = np.nanmean(rank_ic_series) / (np.nanstd(rank_ic_series) + 1e-8) if len(rank_ic_series) > 0 else 0
    
    return {
        'ic': ic,
        'rank_ic': rank_ic,
        'icir': icir,
        'rank_icir': rank_icir
    }

def calculate_ic(pred, target):
    """计算信息系数 (Information Coefficient)"""
    pred = pred.view(-1).detach().cpu().numpy()
    target = target.view(-1).detach().cpu().numpy()
    return np.corrcoef(pred, target)[0, 1]  # Pearson相关系数

def safe_corrcoef(x, y):
    """安全的相关系数计算"""
    with np.errstate(invalid='ignore'):
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return 0.0
        corr = np.corrcoef(x, y)[0, 1]
        return corr if not np.isnan(corr) else 0.0

def safe_spearmanr(x, y):
    """安全的Spearman相关系数计算"""
    with np.errstate(invalid='ignore'):
        if len(x) < 2 or len(y) < 2:
            return 0.0
        corr = spearmanr(x, y).correlation
        return corr if not np.isnan(corr) else 0.0
    
def train_model(model, train_data, val_data, epochs, lr):
    """优化后的训练函数"""
    print("Training...")
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    history = {
        'train_loss': [],
        'val_metrics': [],
        'train_ic': []
    }

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_metrics = {
            'total_loss': 0.0,
            'total_ic': 0.0,
            'num_samples': 0
        }
        
        for batch in train_data:
            x, edge_index, edge_attr, y, fadj, sadj = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
            optimizer.zero_grad()
            
            y_pred = model(x, edge_index)
            # y_pred = model(x, sadj, fadj, edge_index)
            y = y.view(-1, 1)
            
            loss = F.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            
            # 指标记录
            batch_size = y.size(0)
            epoch_metrics['total_loss'] += loss.item() * batch_size
            epoch_metrics['num_samples'] += batch_size
            
            # IC指标计算
            with torch.no_grad():
                batch_ic = calculate_ic(y_pred, y)  # 建议封装成独立函数
                epoch_metrics['total_ic'] += batch_ic * batch_size

        # 计算epoch平均指标
        avg_loss = epoch_metrics['total_loss'] / epoch_metrics['num_samples']
        avg_ic = epoch_metrics['total_ic'] / epoch_metrics['num_samples']
        history['train_loss'].append(avg_loss)
        history['train_ic'].append(avg_ic)
        
        # 验证集评估
        val_metrics = evaluate_model(model, val_data)  # 使用优化后的评估函数
        history['val_metrics'].append(val_metrics)
        
        # 学习率调度
        scheduler.step(val_metrics['val_mse'])  # 根据验证集MSE调整学习率

        # 训练过程打印
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Train IC: {avg_ic:.4f} | "
              f"Val MSE: {val_metrics['val_mse']:.4f} | "
              f"Val MAE: {val_metrics['val_mae']:.4f} | "
              f"Val MAPE: {val_metrics['val_mape']:.2f}%")
    
    return history['train_loss'], history['val_metrics'], history['train_ic']
def evaluate_model(model, val_data):
    """优化后的验证函数"""
    model.eval()
    total = {'mse': 0.0, 'mae': 0.0, 'mape': 0.0, 'samples': 0}
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in val_data:
            x, edge_index, edge_attr, y, fadj, sadj = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
            y_pred = model(x, edge_index)
            # y_pred = model(x, sadj, fadj, edge_index)
            y = y.view(-1, 1)
            
            all_preds.append(y_pred)
            all_targets.append(y)
            
            # 累积损失
            total['mse'] += F.mse_loss(y_pred, y, reduction='sum').item()
            total['mae'] += F.l1_loss(y_pred, y, reduction='sum').item()
            total['samples'] += y.size(0)
            
            # 处理MAPE
            non_zero_mask = y != 0
            if non_zero_mask.any():
                total['mape'] += torch.abs((y[non_zero_mask] - y_pred[non_zero_mask]) / 
                                        y[non_zero_mask]).sum().item() * 100

    # 合并所有结果
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    metrics = {
        'val_mse': total['mse'] / total['samples'],
        'val_mae': total['mae'] / total['samples'],
        'val_rmse': torch.sqrt(torch.tensor(total['mse'] / total['samples'])).item(),
        'val_mape': (total['mape'] / total['samples']) if total['mape'] > 0 else float('nan')
    }
    
    # 重新计算全局IC
    metrics['val_ic'] = calculate_ic(preds, targets)
    
    return metrics

def test_model(model, test_data):
    """优化后的测试函数，包含完整指标输出"""
    print("Testing...")
    model.eval()
    
    # 初始化存储
    all_preds, all_targets = [], []
    total = {
        'mse': 0.0, 
        'mae': 0.0, 
        'mape': 0.0,
        'samples': 0
    }
    
    with torch.no_grad():
        for batch in test_data:
            x, edge_index, edge_attr, y, fadj, sadj = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
            y_pred = model(x, edge_index)
            # y_pred = model(x, sadj, fadj, edge_index)
            y = y.view(-1, 1)
            
            # 累积预测值和目标值
            all_preds.append(y_pred)
            all_targets.append(y)
            
            # 计算常规指标
            total['mse'] += F.mse_loss(y_pred, y, reduction='sum').item()
            total['mae'] += F.l1_loss(y_pred, y, reduction='sum').item()
            total['samples'] += y.size(0)
            
            # 处理MAPE
            non_zero_mask = y != 0
            if non_zero_mask.any():
                total['mape'] += torch.abs((y[non_zero_mask] - y_pred[non_zero_mask]) / 
                                   y[non_zero_mask]).sum().item() * 100

    # 合并所有结果
    preds = torch.cat(all_preds).view(-1)
    targets = torch.cat(all_targets).view(-1)
    
    # 转换为numpy数组
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    metrics_dict = calculate_all_metrics(preds_np, targets_np)
    ic = metrics_dict["ic"]
    rank_ic = metrics_dict["rank_ic"]
    icir = metrics_dict["icir"]
    rank_icir = metrics_dict["rank_icir"]

    # 计算完整指标
    metrics = {
        'test_mse': total['mse'] / total['samples'],
        'test_mae': total['mae'] / total['samples'],
        'test_rmse': torch.sqrt(torch.tensor(total['mse'] / total['samples'])).item(),
        'test_mape': (total['mape'] / total['samples']) if total['mape'] > 0 else float('nan'),
        "ic":ic,
        "rank_ic":rank_ic,
        "icir":icir,
        "rank_icir":rank_icir
    }
    
    # 打印结果
    print(f"""Test Results:
          MSE: {metrics['test_mse']:.4f}
          MAE: {metrics['test_mae']:.4f}
          RMSE: {metrics['test_rmse']:.4f}
          MAPE: {metrics['test_mape']:.2f}%
          IC: {metrics['ic']:.4f}
          Rank IC: {metrics['rank_ic']:.4f}
          ICIR: {metrics['icir']:.4f}
          Rank ICIR: {metrics['rank_icir']:.4f}""")
    
    return preds, targets, metrics

