import unittest
import torch
from models.filters.task_aware_filter import TaskAwareFilter


class TestTaskAwareFilter(unittest.TestCase):
    def test_linear_filter(self):
        filter = TaskAwareFilter(input_dim=768, output_dim=768, architecture='linear')
        input_tensor = torch.randn(2, 10, 768)  # Batch size=2, Seq length=10
        output = filter(input_tensor)
        self.assertEqual(output.shape, (2, 10, 768))

    def test_mlp_filter(self):
        filter = TaskAwareFilter(input_dim=768, output_dim=768, architecture='mlp')
        input_tensor = torch.randn(2, 10, 768)
        output = filter(input_tensor)
        self.assertEqual(output.shape, (2, 10, 768))

    def test_transformer_filter(self):
        filter = TaskAwareFilter(input_dim=768, output_dim=768, architecture='transformer')
        input_tensor = torch.randn(10, 2, 768)  # Seq length=10, Batch size=2
        output = filter(input_tensor)
        self.assertEqual(output.shape, (10, 2, 768))


if __name__ == '__main__':
    unittest.main()
