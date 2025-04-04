#!/bin/bash

echo "🔽 下载缺失的模型文件（请根据需要修改下载链接）"

mkdir -p firsttry_CN/gpt2_baseline/gpt2
mkdir -p firsttry_US/gpt2_baseline/gpt2
mkdir -p firsttry_stocknet/gpt2_baseline/gpt2

# 示例下载命令（请替换为实际下载地址）
# curl -o firsttry_CN/gpt2_baseline/gpt2/pytorch_model.bin https://your-link.com/model1.bin
# curl -o firsttry_US/gpt2_baseline/gpt2/pytorch_model.bin https://your-link.com/model2.bin
# curl -o firsttry_stocknet/gpt2_baseline/gpt2/pytorch_model.bin https://your-link.com/model3.bin

echo "✅ 请根据需要手动修改 download_models.sh 中的下载链接。"
