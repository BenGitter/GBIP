import torch
import torch.nn as nn
import torch.optim as optim

from models.adversarial import AdversarialModel

class AdversarialGame:
    def __init__(self, device):
        super(AdversarialGame, self).__init__()
        self.device = device
        self.model = AdversarialModel(in_features=80).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        self.loss = nn.BCEWithLogitsLoss()
        self.loss_sum = torch.zeros(1).to(self.device)
     
    def get_student_loss(self, in_S):
        self.model.eval()
        n = in_S.shape[0]
        target_T = torch.tensor([1.]).to(self.device)
        with torch.no_grad():
            out_S = self.model(in_S.unsqueeze(0))
        return self.loss(out_S, target_T)

    def update(self, in_S, in_T, threshold=0.2):
        self.model.train()
        input = torch.cat((in_S.unsqueeze(0), in_T.unsqueeze(0)), 0)
        targets = torch.tensor([0.0, 1.0]).to(self.device)
        out = self.model(input)
        loss = self.loss(out, targets)

        if loss > threshold:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach()