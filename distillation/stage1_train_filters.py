import argparse
import torch
from transformers import AutoModel, AutoTokenizer
from models.filters.task_aware_filter import TaskAwareFilter
from utils.helpers import load_data_loader
from utils.config import Config
from torch.utils.data import DataLoader
from accelerate import Accelerator
from distillation.loss_functions import prediction_distillation_loss, layer_distillation_loss
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description="Stage I: Train Task-aware Filters")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    return parser.parse_args()


def train_stage1(config):
    accelerator = Accelerator(fp16=config.mixed_precision == "fp16")

    # 加载教师和学生模型
    teacher_model = AutoModel.from_pretrained(config.teacher_model_name_or_path, output_hidden_states=True).to(
        config.device)
    student_model = AutoModel.from_pretrained(config.student_model_name_or_path, output_hidden_states=True).to(
        config.device)

    teacher_model.eval()
    student_model.eval()

    # 初始化过滤器
    teacher_filters = nn.ModuleList([
        TaskAwareFilter(input_dim=config.teacher_hidden_size, output_dim=config.task_output_size,
                        architecture=config.filter_architecture)
        for _ in range(config.num_layers)
    ]).to(config.device)

    student_filters = nn.ModuleList([
        TaskAwareFilter(input_dim=config.student_hidden_size, output_dim=config.task_output_size,
                        architecture=config.filter_architecture)
        for _ in range(config.num_layers)
    ]).to(config.device)

    # 冻结模型参数
    for param in teacher_model.parameters():
        param.requires_grad = False
    for param in student_model.parameters():
        param.requires_grad = False

    # 优化器仅优化过滤器
    optimizer = torch.optim.AdamW(list(teacher_filters.parameters()) + list(student_filters.parameters()),
                                  lr=config.learning_rate)

    # 数据加载器
    train_dataloader = load_data_loader(config, stage="stage1")

    # 训练循环
    teacher_filters, student_filters, train_dataloader, optimizer = accelerator.prepare(
        teacher_filters, student_filters, train_dataloader, optimizer
    )

    teacher_filters.train()
    student_filters.train()

    for epoch in range(config.stage1_epochs):
        for batch in train_dataloader:
            inputs = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)

            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=inputs, attention_mask=attention_mask)
                student_outputs = student_model(input_ids=inputs, attention_mask=attention_mask)

            # 计算任务损失（例如分类任务的交叉熵）
            task_loss = nn.CrossEntropyLoss()(teacher_outputs.logits, labels)

            # 过滤器损失
            filter_loss = 0
            for k in range(config.num_layers):
                teacher_hidden = teacher_outputs.hidden_states[config.layer_mapping(k)]
                student_hidden = student_outputs.hidden_states[k]

                teacher_filtered = teacher_filters[k](teacher_hidden)
                student_filtered = student_filters[k](student_hidden)

                # 假设任务是分类，使用交叉熵损失
                loss_t = nn.CrossEntropyLoss()(teacher_filtered, labels)
                loss_s = nn.CrossEntropyLoss()(student_filtered, labels)
                filter_loss += loss_t + loss_s

            total_loss = filter_loss / config.num_layers

            optimizer.zero_grad()
            accelerator.backward(total_loss)
            optimizer.step()

        print(f"Stage I Epoch {epoch + 1}/{config.stage1_epochs} - Loss: {total_loss.item()}")

    # 保存过滤器
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(teacher_filters.state_dict(), os.path.join(config.output_dir, "teacher_filters.pth"))
        torch.save(student_filters.state_dict(), os.path.join(config.output_dir, "student_filters.pth"))


if __name__ == "__main__":
    args = parse_args()
    config = Config(args.config)
    train_stage1(config)
