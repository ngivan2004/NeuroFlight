import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QFormLayout, QComboBox, QLineEdit, QPushButton, QDateEdit, QMessageBox, QHBoxLayout
from PyQt5.QtCore import QDate
import torch
import torch.nn.functional as F
import numpy as np
import joblib
import pandas as pd
from model_architecture_double_classification import RegressionNNPrice, ClassificationNNTrend
from datetime import datetime


class PredictionApp:
    def __init__(self):
        self.model1_path = './model1.pth'
        self.model2_path = './model2.pth'
        self.preprocessor_path = 'preprocessor.joblib'
        self.label_encoder_path = 'label_encoder.joblib'

        self.preprocessor = joblib.load(self.preprocessor_path)
        self.label_encoder = joblib.load(self.label_encoder_path)

        input_dim = self.preprocessor.transformers_[0][1].get_feature_names_out(
        ).shape[0] + len(self.preprocessor.transformers_[1][1].mean_)

        self.model1 = RegressionNNPrice(input_dim)
        self.model2 = ClassificationNNTrend(input_dim)

        self.model1.load_state_dict(torch.load(self.model1_path))
        self.model2.load_state_dict(torch.load(self.model2_path))
        self.model1.eval()
        self.model2.eval()

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")
        self.model1.to(self.device)
        self.model2.to(self.device)

    def predict(self, data):
        X = data[['startingAirport', 'destinationAirport', 'segmentsAirlineCode',
                  'travelDuration', 'totalFare', 'daysUntilFlight',
                  'flightDayOfWeek', 'flightDayOfMonth', 'flightMonth']]
        X_preprocessed = self.preprocessor.transform(X)
        if hasattr(X_preprocessed, "toarray"):
            X_preprocessed = X_preprocessed.toarray()
        X_tensor = torch.tensor(
            X_preprocessed.astype(np.float32)).to(self.device)

        with torch.no_grad():
            regression_predictions = self.model1(X_tensor)
            classification_predictions = self.model2(X_tensor)

        regression_predictions = regression_predictions.cpu().numpy().flatten()
        classification_predictions = F.softmax(
            classification_predictions, dim=1).cpu().numpy()

        # Calculate confidence for classification predictions
        classification_confidence = np.max(classification_predictions, axis=1)

        return regression_predictions, classification_predictions, classification_confidence


class FlightPricePredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.app = PredictionApp()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Flight Price Predictor')
        self.setGeometry(100, 100, 400, 400)

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.startingAirport = QComboBox()
        airports = ['ATL', 'DFW', 'DEN', 'ORD', 'LAX', 'CLT', 'MIA', 'JFK',
                    'EWR', 'SFO', 'DTW', 'BOS', 'PHL', 'LGA', 'IAD', 'OAK']
        self.startingAirport.addItems(airports)
        form_layout.addRow('Starting Airport:', self.startingAirport)

        self.destinationAirport = QComboBox()
        self.destinationAirport.addItems(airports)
        form_layout.addRow('Destination Airport:', self.destinationAirport)

        self.segmentsAirlineCode = QComboBox()
        segmentsAirlineCode = {
            'AA': 'American Airlines',
            'AS': 'Alaska Airlines',
            'B6': 'JetBlue Airways',
            'DL': 'Delta Air Lines',
            'F9': 'Frontier Airlines',
            'NK': 'Spirit Airlines',
            'UA': 'United Airlines'
        }
        for code, name in segmentsAirlineCode.items():
            self.segmentsAirlineCode.addItem(f'{code} - {name}', code)
        form_layout.addRow('Segments Airline Code:', self.segmentsAirlineCode)

        self.travelDurationHours = QLineEdit()
        self.travelDurationMinutes = QLineEdit()
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel('Hours:'))
        duration_layout.addWidget(self.travelDurationHours)
        duration_layout.addWidget(QLabel('Minutes:'))
        duration_layout.addWidget(self.travelDurationMinutes)
        form_layout.addRow('Travel Duration:', duration_layout)

        self.totalFare = QLineEdit()
        form_layout.addRow('Total Fare (USD):', self.totalFare)

        self.flightDate = QDateEdit()
        self.flightDate.setDisplayFormat('dd/MM/yyyy')
        self.flightDate.setDateRange(QDate(2024, 4, 16), QDate(2024, 10, 5))
        today = QDate.currentDate()
        if today < QDate(2024, 4, 16):
            today = QDate(2024, 4, 16)
        elif today > QDate(2024, 10, 5):
            today = QDate(2024, 10, 5)
        self.flightDate.setDate(today)
        form_layout.addRow('Flight Date (dd/MM/yyyy):', self.flightDate)

        self.predictButton = QPushButton('Predict')
        self.predictButton.clicked.connect(self.predict)

        layout.addLayout(form_layout)
        layout.addWidget(self.predictButton)

        self.resultLabel = QLabel('')
        layout.addWidget(self.resultLabel)

        self.setLayout(layout)

    def predict(self):
        startingAirport = self.startingAirport.currentText()
        destinationAirport = self.destinationAirport.currentText()
        segmentsAirlineCode = self.segmentsAirlineCode.currentData()
        travelDurationHours = int(self.travelDurationHours.text())
        travelDurationMinutes = int(self.travelDurationMinutes.text())
        travelDuration = travelDurationHours * 60 + travelDurationMinutes
        totalFare = float(self.totalFare.text())
        flightDate = self.flightDate.date().toPyDate()

        daysUntilFlight = (flightDate - datetime.now().date()).days
        flightDayOfWeek = flightDate.weekday()
        flightDayOfMonth = flightDate.day
        flightMonth = flightDate.month

        data = pd.DataFrame({
            'startingAirport': [startingAirport],
            'destinationAirport': [destinationAirport],
            'segmentsAirlineCode': [segmentsAirlineCode],
            'travelDuration': [travelDuration],
            'totalFare': [totalFare],
            'daysUntilFlight': [daysUntilFlight],
            'flightDayOfWeek': [flightDayOfWeek],
            'flightDayOfMonth': [flightDayOfMonth],
            'flightMonth': [flightMonth]
        })

        regression_predictions, classification_predictions, classification_confidence = self.app.predict(
            data)
        # single value prediction for price
        lowestPrice = regression_predictions[0]

        price_trend_class = self.app.label_encoder.inverse_transform(
            [np.argmax(classification_predictions)])[0]
        confidence = classification_confidence[0] * 100

        result_text = (f'Predicted lowestPrice: ${lowestPrice:.2f} USD\n'
                       f'Price Trend: {price_trend_class} (Confidence: {confidence:.2f}%)')
        self.resultLabel.setText(result_text)


def main():
    app = QApplication(sys.argv)
    ex = FlightPricePredictor()
    ex.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
