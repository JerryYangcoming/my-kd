import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import DataCollatorWithPadding
from utils.config import Config


def load_data_loader(config, stage="stage1"):
    # 加载预处理后的数据
    dataset_path = f"./data/processed/{config.task_name}"
    dataset = load_from_disk(dataset_path)

    if stage == "stage1":
        # Stage I: 仅训练过滤器，可能使用部分任务相关的数据
        train_dataset = dataset['train']
    elif stage == "stage2":
        # Stage II: 训练学生模型
        train_dataset = dataset['train']
    else:
        raise ValueError("Unsupported stage")

    data_collator = DataCollatorWithPadding(tokenizer=None, padding='max_length' if config.pad_to_max_length else False,
                                            max_length=config.max_length)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.batch_size,
        collate_fn=data_collator
    )
    return train_dataloader
