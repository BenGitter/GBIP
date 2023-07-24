import torch
import torch.nn as nn

class AdversarialModel(nn.Module):
    def __init__(self, in_features=80):
        super(AdversarialModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            LambdaLayer(lambda x: torch.mean(x, dim=[1,2]))
        )
    
    def forward(self, x):
        return self.model(x)
    
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)