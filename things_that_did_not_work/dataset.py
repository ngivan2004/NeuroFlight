import torch
from torch.utils.data import Dataset, DataLoader


class FlightDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(X_train, X_test, y_train, y_test, batch_size=64):
    train_dataset = FlightDataset(X_train, y_train)
    test_dataset = FlightDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
