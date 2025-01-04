#!/bin/bash

# 设置脚本在遇到错误时立即退出
set -e

# 定义数据目录
DATA_DIR="./data"
GLUE_DIR="$DATA_DIR/glue"
SQUAD_DIR="$DATA_DIR/squad"

# 创建必要的目录
mkdir -p "$GLUE_DIR"
mkdir -p "$SQUAD_DIR"

# 下载所有 GLUE 子任务使用 Huggingface datasets 库
echo "Downloading all GLUE datasets using Huggingface datasets..."
GLUE_TASKS=("cola" "sst2" "mrpc" "sts-b" "qqp" "mnli" "qnli" "rte" "wnli")

for TASK in "${GLUE_TASKS[@]}"; do
    echo "Downloading GLUE task: $TASK"
    python -c "
import os
from datasets import load_dataset

dataset = load_dataset('glue', '$TASK')
save_path = os.path.join('$GLUE_DIR', '$TASK')
dataset.save_to_disk(save_path)
"
    echo "GLUE task '$TASK' downloaded and saved to $GLUE_DIR/$TASK"
done

# 下载 SQuAD 数据集
echo "Downloading SQuAD v1.1 dataset..."
# 确保 SQuAD 目录存在
mkdir -p "$SQUAD_DIR"

# 下载训练集
echo "Downloading SQuAD v1.1 training set..."
wget -O "$SQUAD_DIR/train-v1.1.json" https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json

# 下载开发集
echo "Downloading SQuAD v1.1 development set..."
wget -O "$SQUAD_DIR/dev-v1.1.json" https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

echo "SQuAD v1.1 dataset downloaded and saved to $SQUAD_DIR"

echo "All datasets downloaded successfully."
