import unittest
import torch
import torch.nn as nn
from distillation.loss_functions import prediction_distillation_loss, layer_distillation_loss, total_distillation_loss


class TestDistillationLoss(unittest.TestCase):
    def test_prediction_distillation_loss(self):
        teacher_logits = torch.randn(2, 10)
        student_logits = torch.randn(2, 10)
        temperature = 2.0
        loss = prediction_distillation_loss(teacher_logits, student_logits, temperature)
        self.assertTrue(loss > 0)

    def test_layer_distillation_loss(self):
        teacher_hidden = torch.randn(2, 10, 768)
        student_hidden = torch.randn(2, 10, 768)
        loss = layer_distillation_loss(teacher_hidden, student_hidden)
        self.assertTrue(loss > 0)

    def test_total_distillation_loss(self):
        task_loss = torch.tensor(1.0)
        pred_loss = torch.tensor(0.5)
        layer_loss = torch.tensor(0.3)
        alpha1 = 2.5
        alpha2 = 0.1
        total_loss = total_distillation_loss(task_loss, pred_loss, layer_loss, alpha1, alpha2)
        expected_loss = 1.0 + 2.5 * 0.5 + 0.1 * 0.3
        self.assertAlmostEqual(total_loss.item(), expected_loss)


if __name__ == '__main__':
    unittest.main()
