import torch
import torch.nn as nn
import torch.optim as optim

from models.adversarial import AdversarialModel

class AdversarialGame:
    def __init__(self, device):
        super(AdversarialGame, self).__init__()
        self.device = device
        self.model = AdversarialModel(in_features=80).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.loss = nn.BCEWithLogitsLoss()
        self.loss_sum = torch.zeros(1).to(self.device)
     
    def get_student_loss(self, in_S):
        self.model.eval()
        n = in_S.shape[0]
        targets_T = torch.ones(n).to(self.device)
        # target_T = torch.ones(n).to(self.device)
        # out_S = self.model(in_S.unsqueeze(0)) #.squeeze(dim=1)
        out_S = self.model(in_S)
        return self.loss(out_S, targets_T)

    def update(self, in_S, in_T):
        self.model.train()
        # input = torch.cat((in_S, in_T), 0)
        target_S = torch.zeros(in_S.shape[0]).to(self.device)
        target_T = torch.ones(in_T.shape[0]).to(self.device)
        targets = torch.cat((target_S, target_T))

        input = torch.cat((in_S, in_T), 0)
        # targets = torch.tensor([0., 1.]).to(self.device)

        out = self.model(input) #.squeeze(dim=1)
        loss = self.loss(out, targets)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach()
