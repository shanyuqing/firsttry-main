import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def construct_graph(correlation_matrix):
    
    # 基于相关系数矩阵构建图，相关系数大于阈值的公司之间有边
    num_stocks = correlation_matrix.shape[0]
    edge_index = []
    edge_attr = []
    
    for i in range(num_stocks):
        for j in range(num_stocks):
            # 这里我们保留正负相关的边，只要它们的值不为0
            if correlation_matrix[i, j] != 0:  
                edge_index.append([i, j])
                edge_attr.append(correlation_matrix[i, j])  # 相关系数作为边权重
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_index, edge_attr

def normalize(mx):
    """Row-normalize  matrix"""
    rowsum = np.array(mx.sum(axis = 1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def create_data(stock_data, sadj, time_window, k):
    num_stocks, num_time_steps, num_features = stock_data.shape
    
    # 初始化存储批次数据的列表
    batch_data_list = []

    # 滚动窗口生成批次
    for start in range(num_time_steps - time_window):

        # 当前窗口的节点特征：过去20天的收盘价
        node_features = torch.tensor(stock_data[:, start:start + time_window, 3], dtype=torch.float32)  # 使用收盘价作为节点特征
        node_features = node_features.view(num_stocks, time_window)

        # 创建图的边和边特征
        edge_index, edge_attr = construct_graph(sadj)

        # 目标值：窗口结束后的第一个时间步的收盘价
        y = torch.tensor(stock_data[:, start + time_window, 3], dtype=torch.float32)
        
        # 创建动态特征邻接矩阵
        dynamic_similarity_matrix = cosine_similarity(node_features)
        # 将相似度矩阵转化为 PyTorch 张量
        fadj = torch.tensor(dynamic_similarity_matrix, dtype=torch.float32)
        
        _, indices = fadj.topk(k, dim=1)
        for i in range(fadj.size(0)):
            fadj[i][fadj[i] < fadj[i, indices[i][-1]]] = 0

        # 对邻接矩阵进行对称化处理
        fadj = fadj + fadj.T.mul(fadj.T > fadj) - fadj.mul(fadj.T > fadj)
        # 对 fadj 加上单位矩阵（自环），然后进行归一化
        fadj = normalize(fadj + torch.eye(fadj.size(0), dtype=torch.float32))
        fadj = torch.from_numpy(fadj)
        
        # 将数据添加到批次列表
        batch_data_list.append((node_features.to(device), edge_index.to(device), edge_attr.to(device), y.to(device), fadj.to(device), sadj.to(device)))
        

    # 返回一个包含num_stocks*batch_size个图数据对象的列表
    return batch_data_list


def create_data_for_best_k(stock_data, sadj, time_window, k):
    num_stocks, num_time_steps, num_features = stock_data.shape
    
    # 初始化存储批次数据的列表
    batch_data_list = []

    # 滚动窗口生成批次
    for start in range(num_time_steps - time_window):

        # 当前窗口的节点特征：过去20天的收盘价
        node_features = torch.tensor(stock_data[:, start:start + time_window, 3], dtype=torch.float32)  # 使用收盘价作为节点特征
        node_features = node_features.view(num_stocks, time_window)

        # 创建图的边和边特征
        edge_index, edge_attr = construct_graph(sadj)

        # 目标值：窗口结束后的第一个时间步的收盘价
        y = torch.tensor(stock_data[:, start + time_window, 4], dtype=torch.float32)
        
        # 创建动态特征邻接矩阵
        dynamic_similarity_matrix = cosine_similarity(node_features)
        # 将相似度矩阵转化为 PyTorch 张量
        fadj = torch.tensor(dynamic_similarity_matrix, dtype=torch.float32)

        _, indices = fadj.topk(k, dim=1)
        for i in range(fadj.size(0)):
            fadj[i][fadj[i] < fadj[i, indices[i][-1]]] = 0
        fadj = torch.tensor(fadj, dtype=torch.float32)

        # 对邻接矩阵进行对称化处理
        fadj = fadj + fadj.T.mul(fadj.T > fadj) - fadj.mul(fadj.T > fadj)
        # 对 fadj 加上单位矩阵（自环），然后进行归一化
        fadj = normalize(fadj + torch.eye(fadj.size(0), dtype=torch.float32))
        fadj = torch.from_numpy(fadj)
        
        # 将数据添加到批次列表
        batch_data_list.append((node_features.to(device), edge_index.to(device), edge_attr.to(device), y.to(device), fadj.to(device), sadj.to(device)))
        

    # 返回一个包含num_stocks*batch_size个图数据对象的列表
    return batch_data_list



