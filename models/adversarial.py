import torch.nn as nn

class AdversarialModel(nn.Module):
    def __init__(self, in_features=85):
        super(AdversarialModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        return self.model(x)