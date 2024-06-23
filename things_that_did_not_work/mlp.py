import torch
import torch.nn as nn


class FlightPricePredictor(nn.Module):
    def __init__(self):
        super(FlightPricePredictor, self).__init__()
        self.fc1 = nn.Linear(10, 64)  # 10 input features
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        out = self.fc4(x)
        return out
