import unittest
import torch
from models.student.student_model import StudentModel

class TestStudentModel(unittest.TestCase):
    def test_student_model_forward(self):
        model = StudentModel("bert-base-uncased")
        input_ids = torch.randint(0, 1000, (2, 128))  # Batch size=2, Seq length=128
        attention_mask = torch.ones((2, 128))
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        self.assertTrue(hasattr(outputs, 'hidden_states'))
        self.assertEqual(len(outputs.hidden_states), model.model.config.num_hidden_layers + 1)

if __name__ == '__main__':
    unittest.main()
