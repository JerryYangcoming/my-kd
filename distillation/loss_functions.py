import torch.nn.functional as F
import torch.nn as nn

def prediction_distillation_loss(teacher_logits, student_logits, temperature):
    """
    计算KL散度损失
    """
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return loss

def layer_distillation_loss(teacher_hidden, student_hidden):
    """
    计算MSE损失
    """
    loss = F.mse_loss(student_hidden, teacher_hidden.detach(), reduction='mean')
    return loss

def total_distillation_loss(task_loss, pred_loss, layer_loss, alpha1, alpha2):
    """
    组合总损失
    """
    return task_loss + alpha1 * pred_loss + alpha2 * layer_loss
