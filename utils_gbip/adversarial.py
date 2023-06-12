import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from models.adversarial import AdversarialModel, AdversarialModel2

class AdversarialGame:
    def __init__(self, device):
        super(AdversarialGame, self).__init__()
        self.device = device
        self.model = AdversarialModel(in_features=85).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.loss = nn.BCEWithLogitsLoss()
        self.loss_sum = torch.zeros(1).to(self.device)
     
    def run(self, p_S, p_T):
        n = p_S.shape[0]
        input = torch.cat((p_S.detach().clone(), p_T), 0)
        targets_S = torch.zeros(n).to(self.device)
        targets_T = torch.ones(n).to(self.device)
        targets = torch.cat((targets_S, targets_T))
        
        # loss for Adversarial Model
        # out = self.model(input[:, np.r_[:4, 5:85]]).squeeze(dim=1)
        out = self.model(input).squeeze(dim=1)
        self.loss_sum += 1/3 * self.loss(out, targets)

        # return loss for Student Model
        # out_S = self.model(p_S[:, np.r_[:4, 5:85]]).squeeze(dim=1)
        out_S = self.model(p_S).squeeze(dim=1)
        return self.loss(out_S, targets_T)

    def update(self):
        self.optimizer.zero_grad()
        self.loss_sum.backward()
        self.loss_sum = torch.zeros(1).to(self.device)
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def get_loss_sum(self):
        return self.loss_sum
    
class AdversarialGame2:
    def __init__(self, device):
        super(AdversarialGame2, self).__init__()
        self.device = device
        self.model = AdversarialModel2().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5, weight_decay=1e-4)
        self.loss = nn.BCEWithLogitsLoss()
        self.loss_sum = torch.zeros(1).to(self.device)
     
    def run(self, in_S, in_T):
        input = torch.cat((in_S.detach().clone(), in_T), 0)
        target_S = torch.zeros(in_S.shape[0]).to(self.device)
        target_T = torch.ones(in_T.shape[0]).to(self.device)
        targets = torch.cat((target_S, target_T))
        
        # loss for Adversarial Model
        out = self.model(input).squeeze(dim=1)
        self.loss_sum += self.loss(out, targets)

        # return loss for Student Model
        # out_S = self.model(p_S[:, np.r_[:4, 5:85]]).squeeze(dim=1)
        out_S = self.model(in_S).squeeze(dim=1)
        return self.loss(out_S, target_T)

    def update(self):
        self.optimizer.zero_grad()
        if self.loss_sum > 0.1:
            self.loss_sum.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.loss_sum = torch.zeros(1).to(self.device)
    
    def get_loss_sum(self):
        return self.loss_sum