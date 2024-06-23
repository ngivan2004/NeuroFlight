import torch
import numpy as np
import joblib
import pandas as pd
from model_architecture import RegressionNN


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


def main():
    app = PredictionApp()

    startingAirport = input("Starting Airport: ")
    destinationAirport = input("Destination Airport: ")
    segmentsAirlineCode = input("Segments Airline Code: ")
    travelDuration = int(input("Travel Duration (minutes): "))
    totalFare = float(input("Total Fare: "))
    daysUntilFlight = int(input("Days Until Flight: "))
    flightDayOfWeek = int(input("Flight Day of Week (0=Mon, 6=Sun): "))
    flightDayOfMonth = int(input("Flight Day of Month: "))
    flightMonth = int(input("Flight Month: "))

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

    predictions = app.predict(data)
    result_text = f'Predicted lowestPrice: {predictions[0][0]:.2f}\nPredicted lowestPriceDayLeft: {predictions[0][1]:.2f}'
    print(result_text)


if __name__ == "__main__":
    main()
