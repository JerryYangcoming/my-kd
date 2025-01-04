import torch.nn as nn


class TaskAwareFilter(nn.Module):
    def __init__(self, input_dim, output_dim, architecture='linear'):
        super(TaskAwareFilter, self).__init__()
        if architecture == 'linear':
            self.filter = nn.Linear(input_dim, output_dim)
        elif architecture == 'mlp':
            self.filter = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, output_dim)
            )
        elif architecture == 'transformer':
            self.filter = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4)
        else:
            raise ValueError("Unsupported filter architecture")

    def forward(self, hidden_states):
        return self.filter(hidden_states)
