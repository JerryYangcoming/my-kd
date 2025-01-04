from transformers import AutoModel


class StudentModel(nn.Module):
    def __init__(self, model_name_or_path):
        super(StudentModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
