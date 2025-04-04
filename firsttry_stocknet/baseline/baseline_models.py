import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv
import sys
import os
import numpy as np 
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class GATModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=4):
        super(GATModel, self).__init__()
        
        self.gat1 = GATConv(input_size, hidden_size, heads=num_heads, dropout=0.6)
        self.gat2 = GATConv(hidden_size * num_heads, output_size, heads=1, dropout=0.6)
        
    def forward(self, x, edge_index):
        # First GAT layer
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        
        # Second GAT layer
        x = self.gat2(x, edge_index)
        
        return x

class GRU_GAT_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, num_heads=4):
        super(GRU_GAT_Model, self).__init__()
        
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        
        # GAT layer
        self.gat1 = GATConv(hidden_size, hidden_size, heads=num_heads, dropout=0.6)
        self.gat2 = GATConv(hidden_size * num_heads, output_size, heads=1, dropout=0.6)
        
    def forward(self, x, edge_index):
        # x: Node features (batch_size, seq_len, num_nodes, feature_dim)
        # edge_index: Graph connectivity (edge_index[0], edge_index[1])
        
        # First, process through GRU (Time-series part)
        x, _ = self.gru(x)  # x: (batch_size, seq_len, hidden_size)
       
        # GAT layer
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        
        return x

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        out, _ = self.gru(x)
        # Take the output of the last time step
        out = self.fc(out)  
        return out

class LSTM_GAT_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, num_heads=4):
        super(LSTM_GAT_Model, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        
        # GAT layer
        self.gat1 = GATConv(hidden_size, hidden_size, heads=num_heads, dropout=0.5)
        self.gat2 = GATConv(hidden_size * num_heads, output_size, heads=1, dropout=0.5)
        
    def forward(self, x, edge_index):
        # x: Node features (batch_size, seq_len, num_nodes, feature_dim)
        # edge_index: Graph connectivity (edge_index[0], edge_index[1])
        
        # First, process through LSTM (Time-series part)
        x, _ = self.lstm(x)  
        
        # GAT layer
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        
        return x
    
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = self.fc(out)  
        return out
    
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        out, _ = self.rnn(x)
        # Take the output of the last time step
        out = self.fc(out) 
        return out