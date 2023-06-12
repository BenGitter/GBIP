import torch
import torch.nn as nn

class AdversarialModel(nn.Module):
    def __init__(self, in_features=85):
        super(AdversarialModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.model(x)
    
class AdversarialModel2(nn.Module):
    def __init__(self):
        super(AdversarialModel2, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(255, 512, 3, stride=4),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, stride=5),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        return self.model(x)