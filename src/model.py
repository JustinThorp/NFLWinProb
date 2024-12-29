from torch import nn



class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(8),
            nn.Linear(8, 10),
            nn.ReLU(),
            #nn.Dropout(.5),
            nn.Linear(10, 10),
            nn.ReLU(),
            #nn.Dropout(.5),
            #nn.Linear(10, 25),
            #nn.ReLU(),
            #nn.Dropout(.5),
            #nn.Linear(25, 50),
            #nn.ReLU(),
            #nn.Dropout(.5),
            #nn.Linear(50, 25),
            #nn.ReLU(),
            #nn.Dropout(.5),
            #nn.Linear(25, 10),
            #nn.ReLU(),
            #nn.Dropout(.5),
            nn.Linear(10, 5),
            nn.ReLU(),
            #nn.Dropout(.5),
            nn.Linear(5, 3)
        )

    def forward(self, x):
        x = self.model(x)
        return x[:, :2], x[:, 2:]