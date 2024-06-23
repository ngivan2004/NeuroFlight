import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QFormLayout, QComboBox, QLineEdit, QPushButton, QDateEdit, QMessageBox
from PyQt5.QtCore import QDate
import torch
import numpy as np
import joblib
import pandas as pd
from model_architecture import RegressionNN
from datetime import datetime, timedelta


# tk just wouldnt work so here comes pyqt5

class PredictionApp:
    def __init__(self):
        self.model_path = 'checkpoints/model_epoch_20_83.81742009105184.pth'
        self.preprocessor_path = 'preprocessor.joblib'
        self.preprocessor = joblib.load(self.preprocessor_path)

        input_dim = self.preprocessor.transformers_[0][1].get_feature_names_out(
        ).shape[0] + len(self.preprocessor.transformers_[1][1].mean_)
        self.model = RegressionNN(input_dim)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

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
            predictions = self.model(X_tensor)
        predictions = predictions.cpu().numpy()
        return predictions


class FlightPricePredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.app = PredictionApp()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Flight Price Predictor')
        self.setGeometry(100, 100, 400, 300)

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

        self.segmentsAirlineCode = QLineEdit()
        form_layout.addRow('Segments Airline Code:', self.segmentsAirlineCode)

        self.travelDuration = QLineEdit()
        form_layout.addRow('Travel Duration (minutes):', self.travelDuration)

        self.totalFare = QLineEdit()
        form_layout.addRow('Total Fare:', self.totalFare)

        self.flightDate = QDateEdit()
        self.flightDate.setDisplayFormat('dd/MM/yyyy')
        self.flightDate.setDateRange(QDate(2024, 4, 16), QDate(2024, 10, 5))
        form_layout.addRow('Flight Date:', self.flightDate)

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
        segmentsAirlineCode = self.segmentsAirlineCode.text()
        travelDuration = int(self.travelDuration.text())
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

        predictions = self.app.predict(data)
        lowestPrice = predictions[0][0]
        lowestPriceDayLeft = predictions[0][1]
        lowestPriceDate = flightDate - timedelta(days=int(lowestPriceDayLeft))

        result_text = (f'Predicted lowestPrice: {lowestPrice:.2f}\n'
                       f'Day for Lowest Price: {lowestPriceDate.strftime("%d/%m/%Y")}')
        self.resultLabel.setText(result_text)
        QMessageBox.information(self, 'Prediction Result', result_text)


def main():
    app = QApplication(sys.argv)
    ex = FlightPricePredictor()
    ex.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
