import torch
import torch.nn as nn
import torch.optim as optim

from models.adversarial import AdversarialModel

class NewAdversarial(nn.Module):
    def __init__(self):
        super(NewAdversarial, self).__init__()
        self.mp1 = nn.MaxPool2d(2, 2)
        # self.mp2 = nn.MaxPool2d(10, 10)
        self.conv1 = nn.Conv2d(765, 1024, 3, 2, 1)
        self.conv2 = nn.Conv2d(1024, 512, 3, 2)
        self.conv2
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(32, 1)
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(765, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(512, 1)
        self.model = nn.Sequential(
            nn.Conv2d(765, 1024, 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, 3, padding='same'),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(400, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )
        # print(self.model)

    def forward(self, p):
        bs, _, w, h, _ = p[0].shape
        x1 = self.mp1(self.mp1(p[0].view(bs, -1, w, h)))
        x2 = self.mp1(p[1].view(bs, -1, w//2, h//2))
        x3 = p[2].view(bs, -1, w//4, h//4)
        
        x = torch.cat((x1, x2, x3), 1)
        # # x = self.mp2(x)
        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.mp1(x)
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.mp1(x)
        # x = self.flatten(x)
        # x = self.fc4(x)
        return self.model(x)

class AdversarialGame:
    def __init__(self, device):
        super(AdversarialGame, self).__init__()
        self.device = device
        # self.model = AdversarialModel(in_features=85).to(self.device)
        self.model = NewAdversarial().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-5, weight_decay=1e-4)
        self.loss = nn.BCEWithLogitsLoss()
        self.loss_sum = torch.zeros(1).to(self.device)
     
    def get_student_loss(self, in_S):
        self.model.eval()
        n = in_S.shape[0]
        targets_T = torch.ones(n).to(self.device)
        with torch.no_grad():
            out_S = self.model(in_S).squeeze(dim=1)
        return self.loss(out_S, targets_T)

    def get_student_loss2(self, p):
        self.model.eval()
        bs = p[0].shape[0]
        targets_T = torch.ones(bs).to(self.device)
        out_S = self.model(p)
        return self.loss(out_S.squeeze(), targets_T)

    def update(self, in_S, in_T):
        self.model.train()
        input = torch.cat((in_S, in_T), 0)
        target_S = torch.zeros(in_S.shape[0]).to(self.device)
        target_T = torch.ones(in_T.shape[0]).to(self.device)
        targets = torch.cat((target_S, target_T))

        out = self.model(input).squeeze(dim=1)
        loss = self.loss(out, targets)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach()
    
    def update2(self, p_S, p_T, threshold=0.2):
        self.model.train()
        bs = p_S[0].shape[0]
        targets_S = torch.zeros(bs).to(self.device)
        targets_T = torch.ones(bs).to(self.device)
        targets = torch.cat((targets_S, targets_T))

        # input = torch.cat((in_S, in_T), 0)
        # targets = torch.tensor([0., 1.]).to(self.device)

        out_S = self.model(p_S).squeeze()
        out_T = self.model(p_T).squeeze()
        # out = self.model(input) #.squeeze(dim=1)
        loss = self.loss(torch.cat((out_S, out_T)), targets)

        self.optimizer.zero_grad()
        if loss > threshold:
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.detach()