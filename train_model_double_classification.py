import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

from data_preprocessing_double_classification import load_and_preprocess_data
from model_architecture_double_classification import RegressionNNPrice, ClassificationNNTrend

X_train, X_test, y1_train, y1_test, y2_train, y2_test = load_and_preprocess_data(
    './data/cleanest_iten.csv')


X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

X_train_tensor = torch.tensor(X_train.astype(np.float32))
y1_train_tensor = torch.tensor(y1_train.values.astype(np.float32))
y2_train_tensor = torch.tensor(y2_train.astype(np.int64))
X_test_tensor = torch.tensor(X_test.astype(np.float32))
y1_test_tensor = torch.tensor(y1_test.values.astype(np.float32))
y2_test_tensor = torch.tensor(y2_test.astype(np.int64))

train_dataset1 = TensorDataset(X_train_tensor, y1_train_tensor)
train_dataset2 = TensorDataset(X_train_tensor, y2_train_tensor)
test_dataset1 = TensorDataset(X_test_tensor, y1_test_tensor)
test_dataset2 = TensorDataset(X_test_tensor, y2_test_tensor)

train_loader1 = DataLoader(train_dataset1, batch_size=16384, shuffle=True)
train_loader2 = DataLoader(train_dataset2, batch_size=16384, shuffle=True)
test_loader1 = DataLoader(test_dataset1, batch_size=16384, shuffle=False)
test_loader2 = DataLoader(test_dataset2, batch_size=16384, shuffle=False)

input_dim = X_train_tensor.shape[1]
print("Input Dim: " + str(input_dim))
model1 = RegressionNNPrice(input_dim)
model2 = ClassificationNNTrend(input_dim)

model1.load_state_dict(torch.load('./checkpoints/new3/model1_epoch_60.pth'))
model2.load_state_dict(torch.load('./checkpoints/new3/model2_epoch_60.pth'))


criterion1 = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.01)
optimizer2 = optim.Adam(model2.parameters(), lr=0.01)

scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=10, gamma=0.1)
scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=10, gamma=0.1)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model1.to(device)
model2.to(device)

writer = SummaryWriter()


def calculate_accuracy(predictions, targets):
    _, predicted_classes = torch.max(predictions, 1)
    correct = (predicted_classes == targets).sum().item()
    return correct / len(targets)


os.makedirs('checkpoints', exist_ok=True)

num_epochs = 500
batch_idx = 0  # Global batch index for TensorBoard
for epoch in range(num_epochs):
    model1.train()
    model2.train()
    running_loss1 = 0.0
    running_mae1 = 0.0
    running_r21 = 0.0
    running_loss2 = 0.0
    running_acc2 = 0.0
    for i, ((batch_x1, batch_y1), (batch_x2, batch_y2)) in enumerate(zip(train_loader1, train_loader2)):
        batch_x1, batch_y1 = batch_x1.to(device), batch_y1.to(device)
        batch_x2, batch_y2 = batch_x2.to(device), batch_y2.to(device)

        optimizer1.zero_grad()
        outputs1 = model1(batch_x1)
        loss1 = criterion1(outputs1, batch_y1)
        loss1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        outputs2 = model2(batch_x2)
        loss2 = criterion2(outputs2, batch_y2)
        loss2.backward()
        optimizer2.step()

        running_loss1 += loss1.item()
        running_mae1 += torch.mean(torch.abs(outputs1 - batch_y1)).item()
        running_r21 += 1 - (torch.sum((batch_y1 - outputs1) ** 2) /
                            torch.sum((batch_y1 - torch.mean(batch_y1)) ** 2)).item()

        running_loss2 += loss2.item()
        running_acc2 += calculate_accuracy(outputs2, batch_y2)

        if (i + 1) % 10 == 0:
            batches_left1 = len(train_loader1) - (i + 1)
            batches_left2 = len(train_loader2) - (i + 1)
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}], Model 1 Loss: {loss1.item():.4f}, MAE: {torch.mean(torch.abs(outputs1 - batch_y1)).item():.4f}, R²: {1 - (torch.sum((batch_y1 - outputs1) ** 2) / torch.sum((batch_y1 - torch.mean(batch_y1)) ** 2)).item():.4f}, Batches Left: {batches_left1}')
            print(
                f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}], Model 2 Loss: {loss2.item():.4f}, Accuracy: {calculate_accuracy(outputs2, batch_y2):.4f}, Batches Left: {batches_left2}')

        writer.add_scalar('Loss/train_batch_model1', loss1.item(), batch_idx)
        writer.add_scalar('MAE/train_batch_model1',
                          torch.mean(torch.abs(outputs1 - batch_y1)).item(), batch_idx)
        writer.add_scalar('R2/train_batch_model1', 1 - (torch.sum((batch_y1 - outputs1)
                          ** 2) / torch.sum((batch_y1 - torch.mean(batch_y1)) ** 2)).item(), batch_idx)
        writer.add_scalar('Loss/train_batch_model2', loss2.item(), batch_idx)
        writer.add_scalar('Accuracy/train_batch_model2',
                          calculate_accuracy(outputs2, batch_y2), batch_idx)
        batch_idx += 1

    avg_train_loss1 = running_loss1 / len(train_loader1)
    avg_train_mae1 = running_mae1 / len(train_loader1)
    avg_train_r21 = running_r21 / len(train_loader1)
    avg_train_loss2 = running_loss2 / len(train_loader2)
    avg_train_acc2 = running_acc2 / len(train_loader2)
    writer.add_scalar('Loss/train_epoch_model1', avg_train_loss1, epoch)
    writer.add_scalar('MAE/train_epoch_model1', avg_train_mae1, epoch)
    writer.add_scalar('R2/train_epoch_model1', avg_train_r21, epoch)
    writer.add_scalar('Loss/train_epoch_model2', avg_train_loss2, epoch)
    writer.add_scalar('Accuracy/train_epoch_model2', avg_train_acc2, epoch)

    model1.eval()
    model2.eval()
    test_loss1 = 0.0
    test_mae1 = 0.0
    test_r21 = 0.0
    test_loss2 = 0.0
    test_acc2 = 0.0
    with torch.no_grad():
        for (batch_x1, batch_y1), (batch_x2, batch_y2) in zip(test_loader1, test_loader2):
            batch_x1, batch_y1 = batch_x1.to(device), batch_y1.to(device)
            batch_x2, batch_y2 = batch_x2.to(device), batch_y2.to(device)
            outputs1 = model1(batch_x1)
            outputs2 = model2(batch_x2)
            loss1 = criterion1(outputs1, batch_y1)
            loss2 = criterion2(outputs2, batch_y2)
            test_loss1 += loss1.item()
            test_mae1 += torch.mean(torch.abs(outputs1 - batch_y1)).item()
            test_r21 += 1 - (torch.sum((batch_y1 - outputs1) ** 2) /
                             torch.sum((batch_y1 - torch.mean(batch_y1)) ** 2)).item()
            test_loss2 += loss2.item()
            test_acc2 += calculate_accuracy(outputs2, batch_y2)

    avg_test_loss1 = test_loss1 / len(test_loader1)
    avg_test_mae1 = test_mae1 / len(test_loader1)
    avg_test_r21 = test_r21 / len(test_loader1)
    avg_test_loss2 = test_loss2 / len(test_loader2)
    avg_test_acc2 = test_acc2 / len(test_loader2)
    writer.add_scalar('Loss/test_epoch_model1', avg_test_loss1, epoch)
    writer.add_scalar('MAE/test_epoch_model1', avg_test_mae1, epoch)
    writer.add_scalar('R2/test_epoch_model1', avg_test_r21, epoch)
    writer.add_scalar('Loss/test_epoch_model2', avg_test_loss2, epoch)
    writer.add_scalar('Accuracy/test_epoch_model2', avg_test_acc2, epoch)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss Model 1: {avg_train_loss1:.4f}, Train MAE Model 1: {avg_train_mae1:.4f}, Train R² Model 1: {avg_train_r21:.4f}, Test Loss Model 1: {avg_test_loss1:.4f}, Test MAE Model 1: {avg_test_mae1:.4f}, Test R² Model 1: {avg_test_r21:.4f}')
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss Model 2: {avg_train_loss2:.4f}, Train Accuracy Model 2: {avg_train_acc2:.4f}, Test Loss Model 2: {avg_test_loss2:.4f}, Test Accuracy Model 2: {avg_test_acc2:.4f}')

    torch.save(model1.state_dict(),
               f'./checkpoints/new4/model1_epoch_{epoch+1}.pth')
    torch.save(model2.state_dict(),
               f'./checkpoints/new4/model2_epoch_{epoch+1}.pth')

    scheduler1.step()
    scheduler2.step()

writer.close()
