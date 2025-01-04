#!/bin/bash

# 设置错误处理
set -e

# 定义数据目录
DATA_DIR="./data"
GLUE_DIR="$DATA_DIR/glue"
SQUAD_DIR="$DATA_DIR/squad"

# 创建必要的目录
mkdir -p $GLUE_DIR
mkdir -p $SQUAD_DIR

# 定义要下载的GLUE任务列表
GLUE_TASKS=("cola" "mnli" "qqp" "sst2" "qnli" "rte" "mrpc" "stsb")

# 下载 GLUE 数据集
echo "Downloading all GLUE datasets using Huggingface datasets..."
for TASK in "${GLUE_TASKS[@]}"; do
    echo "Downloading GLUE task: $TASK"
    python3 -c "from datasets import load_dataset; load_dataset('glue', '$TASK', cache_dir='$GLUE_DIR')"
    echo "GLUE task $TASK downloaded successfully."
done

echo "All GLUE datasets have been downloaded and saved to $GLUE_DIR."

# 下载 SQuAD 数据集
echo "Downloading SQuAD dataset..."
mkdir -p $SQUAD_DIR

# 下载训练集
wget -O $SQUAD_DIR/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json

# 下载开发集
wget -O $SQUAD_DIR/dev-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

echo "SQuAD dataset downloaded and saved to $SQUAD_DIR."

# 提示完成
echo "All datasets have been successfully downloaded and preprocessed."
