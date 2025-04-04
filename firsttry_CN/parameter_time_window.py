import subprocess

# 定义超参数搜索范围
time_window = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

# 结果存储文件
output_file = "/root/firsttry-main/firsttry_CN/time_window_results.csv"

# 运行多个超参数实验
with open(output_file, "w") as f:
    f.write("time_window,MSE,MAE,RMSE,MAPE,IC,RankIC,ICIR,RankICIR\n")  # 写入 CSV 头部

for m in time_window:
    # 运行单个实验并传递超参数 m
    process = subprocess.run(["python", "main.py", str(m)], capture_output=True, text=True)

    # 获取实验结果
    output = process.stdout.strip()
    print(f"Experiment with time_window={m} completed. Output: {output}")

    # 解析输出并存入 CSV
    with open(output_file, "a") as f:
        f.write(f"{m},{output}\n")
