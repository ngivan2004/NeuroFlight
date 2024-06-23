import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from tqdm import tqdm
import joblib


def load_and_preprocess_data(file_path):
    # Load data with progress bar
    print("Loading data...")
    data = pd.read_csv(file_path)

    # One-hot encode categorical features
    print("One-hot encoding categorical features...")
    categorical_features = ['startingAirport',
                            'destinationAirport', 'segmentsAirlineCode']
    numeric_features = ['travelDuration', 'totalFare', 'daysUntilFlight',
                        'flightDayOfWeek', 'flightDayOfMonth', 'flightMonth']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ], remainder='passthrough')

    print("Preprocessing data...")
    X = data[categorical_features + numeric_features]
    y = data[['lowestPrice', 'lowestPriceDayLeft']]

    # tqdm progress bar for fitting and transforming data
    with tqdm(total=2, desc="Preprocessing steps") as pbar:
        X_preprocessed = preprocessor.fit_transform(X)
        pbar.update(1)

        # Save the preprocessor to use again in inference (chatgpt taught me this)
        joblib.dump(preprocessor, 'preprocessor.joblib')

        X_train, X_test, y_train, y_test = train_test_split(
            X_preprocessed, y, test_size=0.2, random_state=42)
        pbar.update(1)

    return X_train, X_test, y_train, y_test
