#!/bin/bash

echo "📦 [1/5] 正在清理超大模型文件（使用 git filter-repo）..."

# 确保 git-filter-repo 安装
if ! command -v git-filter-repo &> /dev/null
then
    echo "❌ 未安装 git-filter-repo，请先运行：pip install git-filter-repo"
    exit 1
fi

# 清理三个模型文件
git filter-repo --path firsttry_CN/gpt2_baseline/gpt2/pytorch_model.bin --invert-paths --force
git filter-repo --path firsttry_US/gpt2_baseline/gpt2/pytorch_model.bin --invert-paths --force
git filter-repo --path firsttry_stocknet/gpt2_baseline/gpt2/pytorch_model.bin --invert-paths --force

echo "✅ 模型文件清理完成"

echo "🛡 [2/5] 添加 .gitignore..."

cat <<EOF > .gitignore
# 忽略所有模型二进制文件
**/*.bin
**/*.pt
**/*.ckpt
**/*.pth

# Python缓存和临时文件
__pycache__/
*.log
*.tmp
.ipynb_checkpoints/
EOF

git add .gitignore
git commit -m "Add .gitignore to exclude large model files"

echo "📥 [3/5] 生成模型下载脚本..."

cat <<'EOF' > download_models.sh
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
EOF

chmod +x download_models.sh
git add download_models.sh
git commit -m "Add script for downloading models (bin files not in repo)"

echo "🚀 [4/5] 正在强制推送到远程仓库..."
git push --force

echo "🎉 [5/5] 所有任务完成！"
echo "✅ 仓库已清理干净，模型文件通过 download_models.sh 下载即可。"
