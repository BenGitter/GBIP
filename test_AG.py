import torch
import torch.nn as nn

"""
A. Terrible results.
Fitness: 6.5638e-07 (after one train epoch)
lag = 2.7, lmg= 0.29
-----------------------------------------------------------
Fitness: 0.23967 (after 5 short [ix=800] train epochs)
"""
class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.mp1 = nn.MaxPool2d(2, 2)
        self.mp2 = nn.MaxPool2d(10, 10)
        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(765, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, p):
        bs, _, w, h, _ = p[0].shape
        x1 = self.mp1(self.mp1(self.mp1(p[0].view(bs, -1, w, h))))
        x2 = self.mp1(self.mp1(p[1].view(bs, -1, w//2, h//2)))
        x3 = self.mp1(p[2].view(bs, -1, w//4, h//4))
        
        x = torch.cat((x1, x2, x3), 1)
        x = self.mp2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x
    

"""
B. Also terrible
Fitness: 0.0060346 (after 4 val epochs)
lag = 3.4, lmg= 0.39
"""
class B(nn.Module):
    def __init__(self):
        super(B, self).__init__()
        self.mp1 = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(765, 256, 3, 2, 1)
        self.conv2 = nn.Conv2d(256, 128, 3, 2)
        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU()

        self.fc = nn.Linear(128, 1)

    def forward(self, p):
        bs, _, w, h, _ = p[0].shape
        x1 = self.mp1(self.mp1(p[0].view(bs, -1, w, h)))
        x2 = self.mp1(p[1].view(bs, -1, w//2, h//2))
        x3 = p[2].view(bs, -1, w//4, h//4)
        
        x = torch.cat((x1, x2, x3), 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.mp1(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


"""
C. Still terrible
Fitness: 0.071071 (after one train epoch; no threshold)
Fitness: 0.063606 (after 6 val epochs; threshold = 0.2)
"""
class C(nn.Module):
    def __init__(self, in_features=80):
        super(C, self).__init__()
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
    

"""
D: Still terrible
Fitness: 0.059868 (after 6 val epochs; threshold = 0.2)
"""
class D(nn.Module):
    def __init__(self, in_features=80):
        super(D, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
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


"""
E: Better?
Fitness: 0.15424 (after 7 val epochs; threshold = 0.2)
Fitness: 0.15367 (after 7 val epochs; threshold = 0.0)
"""
class E1(nn.Module):
    def __init__(self, in_features=80):
        super(E1, self).__init__()
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


"""
E: Better?
Fitness: 0.1449 (after 8 val epochs; threshold = 0.0)
Fitness: (; threshold = 0.2; lr=5e-5)
"""
class E2(nn.Module):
    def __init__(self, in_features=85):
        super(E2, self).__init__()
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


"""
F: Slightly worse than None ([0.2487;  0.269])
Fitness: [0.24564;  0.26696] after first two train epochs
Fitness: 0.29648 (after 10 epochs; Fitness None: 0.3006)
"""
class F(nn.Module):
    def __init__(self):
        super(F, self).__init__()
        self.mp1 = nn.MaxPool2d(2, 2)
        self.model = nn.Sequential(
            nn.Conv2d(765, 1024, 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, 3, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 256, 3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(256, 1, 3, padding='same'),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(100, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, p):
        bs, _, w, h, _ = p[0].shape
        x1 = self.mp1(self.mp1(p[0].view(bs, -1, w, h)))
        x2 = self.mp1(p[1].view(bs, -1, w//2, h//2))
        x3 = p[2].view(bs, -1, w//4, h//4)
        
        x = torch.cat((x1, x2, x3), 1)
        return self.model(x)
    

"""
G: Still below None; slightly better than F
Fitness: [0.24495; 0.26888]
Fitness: 0.2975 (after 10 epochs; Fitness None: 0.3006)
Fitness: 0.24569 (after 5 short [ix=800] train epochs)
"""
class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.mp1 = nn.MaxPool2d(2, 2)
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

    def forward(self, p):
        bs, _, w, h, _ = p[0].shape
        x1 = self.mp1(self.mp1(p[0].view(bs, -1, w, h)))
        x2 = self.mp1(p[1].view(bs, -1, w//2, h//2))
        x3 = p[2].view(bs, -1, w//4, h//4)
        
        x = torch.cat((x1, x2, x3), 1)
        return self.model(x)