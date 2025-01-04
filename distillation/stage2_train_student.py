import argparse
import torch
from transformers import AutoModel, AutoTokenizer
from models.filters.task_aware_filter import TaskAwareFilter
from utils.helpers import load_data_loader
from utils.config import Config
from torch.utils.data import DataLoader
from accelerate import Accelerator
from distillation.loss_functions import prediction_distillation_loss, layer_distillation_loss, total_distillation_loss
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description="Stage II: Train Student Model with Task-aware Distillation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    return parser.parse_args()


def train_stage2(config):
    accelerator = Accelerator(fp16=config.mixed_precision == "fp16")

    # 加载教师模型和过滤器
    teacher_model = AutoModel.from_pretrained(config.teacher_model_name_or_path, output_hidden_states=True).to(
        config.device)
    teacher_filters = nn.ModuleList([
        TaskAwareFilter(input_dim=config.teacher_hidden_size, output_dim=config.task_output_size,
                        architecture=config.filter_architecture)
        for _ in range(config.num_layers)
    ]).to(config.device)
    teacher_filters.load_state_dict(torch.load(os.path.join(config.output_dir, "teacher_filters.pth")))
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    for param in teacher_filters.parameters():
        param.requires_grad = False

    # 加载学生模型和过滤器
    student_model = AutoModel.from_pretrained(config.student_model_name_or_path, output_hidden_states=True).to(
        config.device)
    student_filters = nn.ModuleList([
        TaskAwareFilter(input_dim=config.student_hidden_size, output_dim=config.task_output_size,
                        architecture=config.filter_architecture)
        for _ in range(config.num_layers)
    ]).to(config.device)
    student_filters.load_state_dict(torch.load(os.path.join(config.output_dir, "student_filters.pth")))
    student_model.train()
    for param in student_model.parameters():
        param.requires_grad = True
    for param in student_filters.parameters():
        param.requires_grad = True

    # 优化器仅优化学生模型和过滤器
    optimizer = torch.optim.AdamW(list(student_model.parameters()) + list(student_filters.parameters()),
                                  lr=config.learning_rate)

    # 数据加载器
    train_dataloader = load_data_loader(config, stage="stage2")

    # 训练循环
    student_model, student_filters, train_dataloader, optimizer = accelerator.prepare(
        student_model, student_filters, train_dataloader, optimizer
    )

    for epoch in range(config.stage2_epochs):
        for batch in train_dataloader:
            inputs = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)

            teacher_outputs = teacher_model(input_ids=inputs, attention_mask=attention_mask)
            student_outputs = student_model(input_ids=inputs, attention_mask=attention_mask)

            # 任务损失
            task_loss = nn.CrossEntropyLoss()(student_outputs.logits, labels)

            # 预测蒸馏损失
            pred_loss = prediction_distillation_loss(teacher_outputs.logits, student_outputs.logits, config.temperature)

            # 层级蒸馏损失
            layer_loss = 0
            for k in range(config.num_layers):
                teacher_hidden = teacher_outputs.hidden_states[config.layer_mapping(k)]
                student_hidden = student_outputs.hidden_states[k]

                teacher_filtered = teacher_filters[k](teacher_hidden)
                student_filtered = student_filters[k](student_hidden)

                layer_loss += layer_distillation_loss(teacher_filtered, student_filtered)

            total_loss = total_distillation_loss(task_loss, pred_loss, layer_loss, config.alpha1, config.alpha2)

            optimizer.zero_grad()
            accelerator.backward(total_loss)
            optimizer.step()

        print(f"Stage II Epoch {epoch + 1}/{config.stage2_epochs} - Total Loss: {total_loss.item()}")

    # 保存学生模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        student_model.save_pretrained(os.path.join(config.output_dir, "student_model"))
        torch.save(student_filters.state_dict(), os.path.join(config.output_dir, "student_filters_final.pth"))


if __name__ == "__main__":
    args = parse_args()
    config = Config(args.config)
    train_stage2(config)
