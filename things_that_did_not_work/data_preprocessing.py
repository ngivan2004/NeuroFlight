
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from tqdm import tqdm


def preprocess_data(filepath):

    print("Loading data...")
    data = pd.read_csv(filepath)

    label_encoders = {}
    print("Encoding categorical variables...")
    for column in tqdm(['startingAirport', 'destinationAirport', 'segmentsAirlineCode']):
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    print("Processing dates and durations...")
    data['searchDate'] = pd.to_datetime(data['searchDate'], format='%m-%d')
    data['flightDate'] = pd.to_datetime(data['flightDate'], format='%m-%d')
    data['travelDuration'] = data['travelDuration'].apply(
        lambda x: int(x[2:].replace('H', '').replace('M', '')))

    print("Extracting features...")
    data['searchDayOfWeek'] = data['searchDate'].dt.dayofweek
    data['flightDayOfWeek'] = data['flightDate'].dt.dayofweek
    data['searchMonth'] = data['searchDate'].dt.month
    data['flightMonth'] = data['flightDate'].dt.month
    data['searchDayOfYear'] = data['searchDate'].dt.dayofyear
    data['flightDayOfYear'] = data['flightDate'].dt.dayofyear

    data.drop(columns=['searchDate', 'flightDate'], inplace=True)

    features = ['startingAirport', 'destinationAirport', 'travelDuration', 'totalFare', 'segmentsAirlineCode',
                'daysUntilFlight', 'searchDayOfWeek', 'flightDayOfWeek', 'searchMonth', 'flightMonth', 'searchDayOfYear', 'flightDayOfYear']
    target = ['lowestPrice', 'lowestPriceDayLeft']

    print("Splitting data...")
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print("Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Converting to tensors...")
    X_train = torch.tensor(
        X_train, dtype=torch.float32).reshape(-1, 1, len(features))
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    X_test = torch.tensor(
        X_test, dtype=torch.float32).reshape(-1, 1, len(features))
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    return X_train, X_test, y_train, y_test, features
