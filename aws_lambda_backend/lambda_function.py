import os
import torch
import torch.nn.functional as F
import numpy as np
import joblib
import pandas as pd
from model_architecture_double_classification import RegressionNNPrice, ClassificationNNTrend
from datetime import datetime
import json


base_path = os.path.dirname(os.path.abspath(__file__))


model1_path = os.path.join(base_path, 'models', 'model1.pth')
model2_path = os.path.join(base_path, 'models', 'model2.pth')
preprocessor_path = os.path.join(base_path, 'preprocessor.joblib')
label_encoder_path = os.path.join(base_path, 'label_encoder.joblib')


preprocessor = joblib.load(preprocessor_path)
label_encoder = joblib.load(label_encoder_path)

input_dim = preprocessor.transformers_[0][1].get_feature_names_out(
).shape[0] + len(preprocessor.transformers_[1][1].mean_)

model1 = RegressionNNPrice(input_dim)
model2 = ClassificationNNTrend(input_dim)

model1.load_state_dict(torch.load(
    model1_path, map_location=torch.device('cpu')))
model2.load_state_dict(torch.load(
    model2_path, map_location=torch.device('cpu')))
model1.eval()
model2.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device)
model2.to(device)


def lambda_handler(event, context):
    try:
        # Handle CORS preflight request (i hate this so much aws why do you overwrite everything)
        if event['httpMethod'] == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST,OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                'body': ''
            }

        print("Dump:", json.dumps(event))

        if 'body' in event:
            print("It has a body")
            input_data = json.loads(event["body"])
        else:
            print("It is a nobody.")
            input_data = event

        startingAirport = input_data.get('startingAirport')
        destinationAirport = input_data.get('destinationAirport')
        segmentsAirlineCode = input_data.get('segmentsAirlineCode')
        travelDurationHours = int(input_data.get('travelDurationHours', 0))
        travelDurationMinutes = int(input_data.get('travelDurationMinutes', 0))
        travelDuration = travelDurationHours * 60 + travelDurationMinutes
        totalFare = float(input_data.get('totalFare', 0.0))
        flightDateStr = input_data.get('flightDate')

        if not all([startingAirport, destinationAirport, segmentsAirlineCode, flightDateStr]):
            raise ValueError("Missing required input parameters")

        flightDate = datetime.strptime(flightDateStr, '%d/%m/%Y').date()

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

        X = data[['startingAirport', 'destinationAirport', 'segmentsAirlineCode',
                  'travelDuration', 'totalFare', 'daysUntilFlight',
                  'flightDayOfWeek', 'flightDayOfMonth', 'flightMonth']]
        X_preprocessed = preprocessor.transform(X)

        # sparse matrix dense matrix thing idk
        if hasattr(X_preprocessed, "toarray"):
            X_preprocessed = X_preprocessed.toarray()
        X_tensor = torch.tensor(X_preprocessed.astype(np.float32)).to(device)

        with torch.no_grad():
            regression_predictions = model1(X_tensor)
            classification_predictions = model2(X_tensor)

        regression_predictions = regression_predictions.cpu().numpy().flatten()
        classification_predictions = F.softmax(
            classification_predictions, dim=1).cpu().numpy()

        classification_confidence = np.max(classification_predictions, axis=1)

        lowestPrice = regression_predictions[0]

        price_trend_class = label_encoder.inverse_transform(
            [np.argmax(classification_predictions)])[0]
        confidence = classification_confidence[0] * 100

        result = {
            'predictedLowestPrice': f'{lowestPrice:.2f}',
            'priceTrend': price_trend_class,
            'confidence': f'{confidence:.2f}'
        }

        print("Prediction result:", result)

        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST,OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Content-Type': 'application/json'
            },
            'body': json.dumps(result)
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            # All CORS stuff will get overwritten by aws anw
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST,OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'error': str(e)})
        }
