import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

from data_preprocessing import load_and_preprocess_data
from model_architecture import RegressionNN

X_train, X_test, y_train, y_test = load_and_preprocess_data(
    './data/cleaner_iten.csv')

# Convert sparse matrices to dense because somehow pytorch doesnt like sparse matrices ????
X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

X_train_tensor = torch.tensor(X_train.astype(np.float32))
y_train_tensor = torch.tensor(y_train.values.astype(np.float32))
X_test_tensor = torch.tensor(X_test.astype(np.float32))
y_test_tensor = torch.tensor(y_test.values.astype(np.float32))

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

input_dim = X_train_tensor.shape[1]
print("Input Dim: "+str(input_dim))
model = RegressionNN(input_dim)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01)  # Using AdamW optimizer

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

writer = SummaryWriter()


def calculate_accuracy(predictions, targets):
    mae = torch.mean(torch.abs(predictions - targets), dim=0)
    accuracy = 100 - mae
    return accuracy


os.makedirs('checkpoints', exist_ok=True)


num_epochs = 20
batch_idx = 0  # Global batch index for TensorBoard
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, batch_y).mean().item()

        if (i + 1) % 100 == 0:
            batches_left = len(train_loader) - (i + 1)
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}], Loss: {loss.item():.4f}, Accuracy: {calculate_accuracy(outputs, batch_y).mean().item():.2f}%, Batches Left: {batches_left}')

        writer.add_scalar('Loss/train_batch', loss.item(), batch_idx)
        writer.add_scalar('Accuracy/train_batch',
                          calculate_accuracy(outputs, batch_y).mean().item(), batch_idx)
        batch_idx += 1

    avg_train_loss = running_loss / len(train_loader)
    avg_train_accuracy = running_accuracy / len(train_loader)
    writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
    writer.add_scalar('Accuracy/train_epoch', avg_train_accuracy, epoch)

    # torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pth')

    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            test_accuracy += calculate_accuracy(outputs, batch_y).mean().item()

    avg_test_loss = test_loss / len(test_loader)
    avg_test_accuracy = test_accuracy / len(test_loader)
    writer.add_scalar('Loss/test_epoch', avg_test_loss, epoch)
    writer.add_scalar('Accuracy/test_epoch', avg_test_accuracy, epoch)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.2f}%, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.2f}%')

    torch.save(model.state_dict(),
               f'checkpoints/model_epoch_{epoch+1}_{avg_test_accuracy}.pth')

    scheduler.step()


writer.close()
