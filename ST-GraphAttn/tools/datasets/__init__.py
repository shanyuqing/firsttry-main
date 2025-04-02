# 待修改
from .data_utils import construct_graph, normalize, create_data, create_data_for_best_k
from torch.utils.data import DataLoader
import os 
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

__all__ = {
    'construct_graph': construct_graph,
    'normalize': normalize,
    'create_data': create_data,
    "create_data_for_best_k": create_data_for_best_k  
}

def build_dataloader(cfg, device):
    dataset_name = cfg.DATA_CONFIG.DATASET   # 获取数据集名称
    data_path = cfg.DATA_CONFIG.DATA_PATH + dataset_name
    stuc_matrix = cfg.DATA_CONFIG.MATRIX_FILE #获取结构依赖相关性系数
    struc_matrix_path = cfg.DATA_CONFIG.DATA_PATH +  stuc_matrix
    batch_size = cfg.OPTIMIZATION.BATCH_SIZE
    k = cfg.DATA_CONFIG.K
    # dataset = __all__[dataset_name]()  # 根据数据集名称获取数据集类
    # dataset.loaddata(data_path, device)  # 加载数据集

    file_list = [f for f in os.listdir(data_path) if f.endswith('.txt')]
    all_data = pd.DataFrame()

    for file_name in file_list:
        file_path = os.path.join(data_path, file_name)
        df = pd.read_csv(file_path, sep='\t', header=None, names=['dt', 'open', 'high', 'low', 'close', 'adj close', 'volume'])  # 读取文件
        
        # 按行倒序  只有StockNet需要倒序，其他数据集不需要
        df = df.iloc[::-1].reset_index(drop=True)
        df['code'] = file_name[0:-4]  # 在DataFrame中添加一列，用于存储文件名
        all_data = pd.concat([all_data, df], ignore_index=True)  # 合并当前文件数据到总数据

    stock_data = np.zeros((cfg.MODEL.NUM_NODES, cfg.DATA_CONFIG.NUM_DAYS, 6))         

    # 填充数据
    for i, stock_code in enumerate(all_data['code'].unique()):
        stock_df = all_data[all_data['code'] == stock_code]
        # 存储数值特征
        stock_data[i] = stock_df.iloc[:, 1:-1].values  # -1排除code列，1:-1取数值列

    # 数据归一化(整体进行归一)
    normalized_data = np.zeros_like(stock_data)

    for features_index in range(stock_data.shape[2]):  # 从1开始，跳过时间戳列
        stock_type_data = stock_data[:, :, features_index]
        global_min = np.min(stock_type_data)
        global_max = np.max(stock_type_data)
        normalized_data[:, :, features_index] = (stock_type_data - global_min) / (global_max - global_min)
    stock_data = normalized_data

    # 获取结构邻接矩阵
    correlation_matrix = pd.read_csv(struc_matrix_path)
    sadj = correlation_matrix.iloc[0:cfg.MODEL.NUM_NODES, 1:(cfg.MODEL.NUM_NODES+1)].values
    sadj[abs(sadj) < cfg.DATA_CONFIG.STRUC_THRESHOLD] = 0  
    sadj = torch.tensor(sadj, dtype=torch.float32) 
    sadj = sadj + sadj.T.mul(sadj.T > sadj) - sadj.mul(sadj.T > sadj)
    sadj = normalize(sadj + torch.eye(sadj.size(0), dtype=torch.float32))
    sadj = torch.from_numpy(sadj)

    # 创建训练数据集和测试集
    batch_data_list = create_data(stock_data, sadj, time_window= cfg.MODEL.INPUT_DIM, k=k)
    train_data, temp_data = train_test_split(batch_data_list, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    return train_data, val_data, test_data

