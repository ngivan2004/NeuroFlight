import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, r2_score
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from data_preprocessing_double_classification import load_and_preprocess_data
from model_architecture_double_classification import RegressionNNPrice, ClassificationNNTrend

X_train, X_test, y1_train, y1_test, y2_train, y2_test = load_and_preprocess_data(
    './data/cleanest_iten.csv')

X_test = X_test[:300]
y1_test = y1_test[:300]
y2_test = y2_test[:300]

X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

X_test_tensor = torch.tensor(X_test.astype(np.float32))
y1_test_tensor = torch.tensor(y1_test.values.astype(np.float32))
y2_test_tensor = torch.tensor(y2_test.astype(np.int64))

input_dim = X_test_tensor.shape[1]
model1 = RegressionNNPrice(input_dim)
model2 = ClassificationNNTrend(input_dim)

model1.load_state_dict(torch.load('./model1.pth'))
model2.load_state_dict(torch.load('./model2.pth'))

device = torch.device("mps" if torch.backends.mps.is_available(
) else "cuda" if torch.cuda.is_available() else "cpu")
model1.to(device)
model2.to(device)

batch_size = 2048
test_dataset = TensorDataset(X_test_tensor, y1_test_tensor, y2_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model1.eval()
model2.eval()

y1_preds = []
y2_preds_classes = []
y1_tests = []
y2_tests = []

with torch.no_grad():
    for batch in test_loader:
        X_batch, y1_batch, y2_batch = batch
        X_batch = X_batch.to(device)
        y1_batch = y1_batch.to(device)
        y2_batch = y2_batch.to(device)

        y1_pred = model1(X_batch)
        y1_preds.extend(y1_pred.cpu().numpy())
        y1_tests.extend(y1_batch.cpu().numpy())

        y2_pred = model2(X_batch)
        _, y2_pred_classes = torch.max(y2_pred, 1)
        y2_preds_classes.extend(y2_pred_classes.cpu().numpy())
        y2_tests.extend(y2_batch.cpu().numpy())


y1_preds = np.array(y1_preds)
y1_tests = np.array(y1_tests)
y2_preds_classes = np.array(y2_preds_classes)
y2_tests = np.array(y2_tests)


plt.figure(figsize=(10, 5))
plt.scatter(y1_tests, y1_preds, alpha=0.3)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Regression Model: Predicted vs Actual Prices')
plt.plot([y1_tests.min(), y1_tests.max()], [
         y1_tests.min(), y1_tests.max()], 'k--', lw=2)
plt.grid(True)
plt.show()


r2 = r2_score(y1_tests, y1_preds)
print(f'Regression model R^2: {r2:.4f}')


categories = ["Not yet lowest", "Already lowest"]
label_encoder = joblib.load('label_encoder.joblib')
y2_test_labels = label_encoder.inverse_transform(y2_tests)
y2_pred_labels = label_encoder.inverse_transform(y2_preds_classes)

accuracy = accuracy_score(y2_test_labels, y2_pred_labels) * 100
print(f'Overall classification accuracy: {accuracy:.2f}%')


category_accuracy = {}
for category in categories:
    idx = y2_test_labels == category
    category_accuracy[category] = accuracy_score(
        y2_test_labels[idx], y2_pred_labels[idx]) * 100

plt.figure(figsize=(10, 5))
plt.bar(category_accuracy.keys(), category_accuracy.values())
plt.xlabel('Category')
plt.ylabel('Accuracy (%)')
plt.title('Classification Model: Accuracy per Category')
plt.ylim(0, 100)
plt.grid(True)
plt.show()


confusion_matrix = pd.crosstab(y2_test_labels, y2_pred_labels, rownames=[
                               'Actual'], colnames=['Predicted'])


print(confusion_matrix)
