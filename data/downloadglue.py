import os
from datasets import load_dataset

# 定义 GLUE 子任务列表
GLUE_TASKS = ["cola", "sst2", "mrpc", "sts-b", "qqp", "mnli", "qnli", "rte", "wnli"]

# 定义保存路径
GLUE_DIR = "./glue"

# 创建保存目录
os.makedirs(GLUE_DIR, exist_ok=True)

for task in GLUE_TASKS:
    print(f"Downloading GLUE task: {task}")
    try:
        # 移除 download_mode 参数
        dataset = load_dataset('glue', task)
        save_path = os.path.join(GLUE_DIR, task)
        dataset.save_to_disk(save_path)
        print(f"GLUE task '{task}' downloaded and saved to {save_path}")
    except Exception as e:
        print(f"Failed to download GLUE task '{task}': {e}")
