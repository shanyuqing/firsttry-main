# ✅ 修改后的训练、验证和测试过程，结构与 GCN 代码一致
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr 

def calculate_metrics(list1, list2):
    values1 = [t.item() for t in list1]
    values2 = [t.item() for t in list2]

    if np.allclose(values1, values2):
        return 1.0, 1.0, 1.0, 1.0

    ic = np.corrcoef(values1, values2)[0, 1] if np.std(values1) != 0 else 0
    if np.isnan(ic):
        ic = 0

    rank_ic, _ = spearmanr(values1, values2)
    if np.isnan(rank_ic):
        rank_ic = 0

    ic_std = np.std(np.array(values1) - np.array(values2))
    rank_ic_std = np.std(np.argsort(values1) - np.argsort(values2))

    icir = ic / ic_std if ic_std != 0 else 0
    rankicir = rank_ic / rank_ic_std if rank_ic_std != 0 else 0

    return ic, rank_ic, icir, rankicir

# ✅ 训练过程

def train_model(model, optimizer, train_loader, val_loader, epochs):
    loss_values, val_loss_values, IC_values = [], [], []
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_ic = 0.0, []

        for X_seq, X_llm, y in train_loader:
            X_seq, X_llm, y = X_seq.to(device), X_llm.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X_seq, X_llm)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            ic, rank_ic, icir, rankicir = calculate_metrics(y_pred, y)
            if not np.isnan(ic):
                epoch_ic.append(ic)
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_loss = evaluate_model(model, val_loader)

        loss_values.append(avg_loss)
        val_loss_values.append(val_loss)
        IC_values.append(np.nanmean(epoch_ic))

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, IC: {IC_values[-1]:.4f}")

    return loss_values, val_loss_values, IC_values

# ✅ 验证过程

def evaluate_model(model, val_loader):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for X_seq, X_llm, y in val_loader:
            X_seq, X_llm, y = X_seq.to(device), X_llm.to(device), y.to(device)
            y_pred = model(X_seq, X_llm)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# ✅ 测试过程

def test_model(model, test_loader):
    model.eval()
    pred_list, target_list = [], []
    mse_total, mae_total, rmse_total, mape_total = 0.0, 0.0, 0.0, 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for X_seq, X_llm, y in test_loader:
            X_seq, X_llm, y = X_seq.to(device), X_llm.to(device), y.to(device)
            y_pred = model(X_seq, X_llm)
            pred_list.append(y_pred)
            target_list.append(y)

            mse = criterion(y_pred, y)
            mae = nn.L1Loss()(y_pred, y)
            rmse = torch.sqrt(mse)
            mask = y != 0
            mape = (torch.abs((y[mask] - y_pred[mask]) / (y[mask] + 1e-8))).mean() * 100

            mse_total += mse.item()
            mae_total += mae.item()
            rmse_total += rmse.item()
            mape_total += mape.item()

    pred_all = torch.cat(pred_list, dim=0)
    target_all = torch.cat(target_list, dim=0)

    return pred_all, target_all, mse_total / len(test_loader), mae_total / len(test_loader), rmse_total / len(test_loader), mape_total / len(test_loader)

# ✅ DataLoader 构造方法

def build_dataloader(X_seq, X_llm, y, batch_size=64):
    dataset = torch.utils.data.TensorDataset(X_seq, X_llm, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

import os
import re
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载本地模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import stock_data  # stock_data: (81, 1257, 6)

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "gpt2_baseline/gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model_gpt2 = GPT2LMHeadModel.from_pretrained(model_path).to(device)
model_gpt2.eval()

# 准备数据集
X_seq_list, X_llm_list, y_list = [], [], []
stock_prices = stock_data[:, :, 3]  # 收盘价

num_failed = 0
total_samples = 0

for company_idx in range(stock_prices.shape[0]):
    for i in range(stock_prices.shape[1] - 12):
        seq_data = stock_prices[company_idx, i:i+12]
        actual_next_value = stock_prices[company_idx, i+12]

        prompt = f"The stock price over the past 12 days was {', '.join(map(str, seq_data))}. What is the stock price going to be tomorrow?"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=60).to(device)

        with torch.no_grad():
            outputs = model_gpt2.generate(
                **inputs,
                max_length=80,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1
            )

        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        match = re.search(r'\d+\.\d+', predicted_text)
        predicted_value = float(match.group()) if match else np.nan

        if np.isnan(predicted_value):
            num_failed += 1
            continue

        X_seq_list.append(seq_data)
        X_llm_list.append(predicted_value)
        y_list.append(actual_next_value)
        total_samples += 1

print(f"\n❗GPT2 生成缺失率: {num_failed / (total_samples + num_failed) * 100:.2f}%")

# 转换为张量
X_seq = np.array(X_seq_list).reshape(-1, 12, 1)  # LSTM 输入
X_llm = np.array(X_llm_list).reshape(-1, 1)      # GPT 预测值
y = np.array(y_list)

# 划分数据集
X_train_seq, X_temp_seq, X_train_llm, X_temp_llm, y_train, y_temp = train_test_split(
    X_seq, X_llm, y, test_size=0.4, random_state=42
)
X_val_seq, X_test_seq, X_val_llm, X_test_llm, y_val, y_test = train_test_split(
    X_temp_seq, X_temp_llm, y_temp, test_size=0.5, random_state=42
)

# 转换为 Tensor
X_train_seq = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
X_train_llm = torch.tensor(X_train_llm, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

X_val_seq = torch.tensor(X_val_seq, dtype=torch.float32).to(device)
X_val_llm = torch.tensor(X_val_llm, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

X_test_seq = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
X_test_llm = torch.tensor(X_test_llm, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# ✅ 构建融合模型：LSTM + GPT预测值
class LSTMWithGPT(nn.Module):
    def __init__(self):
        super(LSTMWithGPT, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, dropout=0.3, batch_first=True)
        self.llm_fc = nn.Linear(1, 13)
        self.combined_fc = nn.Linear(50 + 13, 1)

    def forward(self, x_seq, x_llm):
        lstm_out, _ = self.lstm(x_seq)
        h_seq = lstm_out[:, -1, :]
        h_llm = torch.relu(self.llm_fc(x_llm))
        h_combined = torch.cat([h_seq, h_llm], dim=1)
        return self.combined_fc(h_combined)

model = LSTMWithGPT().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = build_dataloader(X_train_seq, X_train_llm, y_train)
val_loader = build_dataloader(X_val_seq, X_val_llm, y_val)
test_loader = build_dataloader(X_test_seq, X_test_llm, y_test)
loss, val_loss, ic_list = train_model(model, optimizer, train_loader, val_loader, epochs=200)
pred, target, mse, mae, rmse, mape = test_model(model, test_loader)

ic, rank_ic, icir, rankicir = calculate_metrics(pred, target)
print(f"{mse},{mae},{rmse},{mape},{ic},{rank_ic},{icir},{rankicir}")
