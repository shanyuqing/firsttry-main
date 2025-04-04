import os
import re
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载本地模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import stock_data 

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
    for i in range(stock_prices.shape[1] - 18):
        seq_data = stock_prices[company_idx, i:i+18]
        actual_next_value = stock_prices[company_idx, i+18]

        prompt = f"The stock price over the past 18 days was {', '.join(map(str, seq_data))}. What is the stock price going to be tomorrow?"
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
X_seq = np.array(X_seq_list).reshape(-1, 18, 1)  # LSTM 输入
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
        self.llm_fc = nn.Linear(1, 19)
        self.combined_fc = nn.Linear(50 + 19, 1)

    def forward(self, x_seq, x_llm):
        lstm_out, _ = self.lstm(x_seq)
        h_seq = lstm_out[:, -1, :]
        h_llm = torch.relu(self.llm_fc(x_llm))
        h_combined = torch.cat([h_seq, h_llm], dim=1)
        return self.combined_fc(h_combined)

# 训练模型
model = LSTMWithGPT().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 40

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_seq, X_train_llm)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        model.eval()
        val_loss = criterion(model(X_val_seq, X_val_llm), y_val).item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

# 测试模型
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test_seq, X_test_llm)

# 指标评估
y_pred_test_np = y_pred_test.cpu().numpy()
y_test_np = y_test.cpu().numpy()

mse = mean_squared_error(y_test_np, y_pred_test_np)
mae = mean_absolute_error(y_test_np, y_pred_test_np)
rmse = np.sqrt(mse)
# 替换旧的 MAPE 计算方式
non_zero_mask = y_test_np != 0
mape = np.mean(np.abs((y_test_np[non_zero_mask] - y_pred_test_np[non_zero_mask]) / y_test_np[non_zero_mask])) * 100

print(f"\n✅ 测试完成！")
print(f"MSE  : {mse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.4f}%")

# LSTM 提取了15天价格的时间序列特征
# GPT2 基于自然语言 prompt 提前预测未来价格（作为先验提示）
#模型融合这两者信息，进行最终股价回归预测

# #✅ 测试完成！
# MSE  : 0.0003
# MAE  : 0.0121
# RMSE : 0.0186
# MAPE : 119.8570%