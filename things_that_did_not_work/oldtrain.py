import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_preprocessing import preprocess_data
from dataset import get_dataloaders
from model import FlightPricePredictor

FILEPATH = './data/clean_iten.csv'
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

print("Preprocessing data...")
X_train, X_test, y_train, y_test, features = preprocess_data(FILEPATH)

train_loader, test_loader = get_dataloaders(
    X_train, X_test, y_train, y_test, batch_size=BATCH_SIZE)

model = FlightPricePredictor().to(DEVICE)

criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

writer = SummaryWriter()


def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct_predictions += torch.sum(torch.abs(outputs - y_batch)
                                         < 0.1).item() / y_batch.numel()
        total_predictions += 1

        if batch_idx % 100 == 0:
            accuracy = (correct_predictions / total_predictions) * 100
            print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
            writer.add_scalar('Training/Loss', loss.item(),
                              epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Training/Accuracy', accuracy,
                              epoch * len(train_loader) + batch_idx)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = (correct_predictions / total_predictions) * 100
    print(
        f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%")
    writer.add_scalar('Training/Epoch_Loss', epoch_loss, epoch)
    writer.add_scalar('Training/Epoch_Accuracy', epoch_accuracy, epoch)


def evaluate(model, test_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item()
            correct_predictions += torch.sum(
                torch.abs(outputs - y_batch) < 0.1).item() / y_batch.numel()
            total_predictions += 1

            if batch_idx % 100 == 0:
                accuracy = (correct_predictions / total_predictions) * 100
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(test_loader)}], "
                      f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
                writer.add_scalar('Evaluation/Loss', loss.item(),
                                  epoch * len(test_loader) + batch_idx)
                writer.add_scalar('Evaluation/Accuracy', accuracy,
                                  epoch * len(test_loader) + batch_idx)

    epoch_loss = running_loss / len(test_loader)
    epoch_accuracy = (correct_predictions / total_predictions) * 100
    print(
        f"Evaluation Loss: {epoch_loss:.4f}, Evaluation Accuracy: {epoch_accuracy:.2f}%")
    writer.add_scalar('Evaluation/Epoch_Loss', epoch_loss, epoch)
    writer.add_scalar('Evaluation/Epoch_Accuracy', epoch_accuracy, epoch)


for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    train(model, train_loader, criterion, optimizer, DEVICE, epoch)
    evaluate(model, test_loader, criterion, DEVICE, epoch)
    scheduler.step()

torch.save(model.state_dict(), 'flight_price_predictor.pth')
print("Model training complete and saved as 'flight_price_predictor.pth'")

writer.close()
