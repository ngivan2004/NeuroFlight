import torch
import torch.nn as nn


class FlightPricePredictor(nn.Module):
    def __init__(self):
        super(FlightPricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=12, hidden_size=128,
                            num_layers=3, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)  # Change to single output
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h_0 = torch.zeros(3, x.size(0), 128).to(x.device)
        c_0 = torch.zeros(3, x.size(0), 128).to(x.device)
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        x = self.relu(lstm_out[:, -1, :])
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        out = self.fc2(x)
        return out


class DaysLeftPredictor(nn.Module):
    def __init__(self):
        super(DaysLeftPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=12, hidden_size=128,
                            num_layers=3, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)  # Change to single output
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h_0 = torch.zeros(3, x.size(0), 128).to(x.device)
        c_0 = torch.zeros(3, x.size(0), 128).to(x.device)
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        x = self.relu(lstm_out[:, -1, :])
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        out = self.fc2(x)
        return out
