from torch import nn

class MLP(nn.Module):
    def __init__(self, n_in, n_out, n_hidden):
        super().__init__()

        self.fc_in = nn.Linear(n_in, n_hidden)
        self.fc_out = nn.Linear(n_hidden, n_out)
    
    def forward(self, x):
        x = self.fc_in(x)
        x = self.fc_out(x)
        return x