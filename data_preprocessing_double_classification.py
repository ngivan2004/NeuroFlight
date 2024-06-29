import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from tqdm import tqdm
import joblib


def load_and_preprocess_data(file_path):

    print("Loading data...")
    data = pd.read_csv(file_path)

    def classify_lowest_price(row):
        if row['lowestPriceDayLeft'] < row['daysUntilFlight']:
            return "Not yet lowest"
        else:
            return "Already lowest"

    data['price_trend'] = data.apply(classify_lowest_price, axis=1)

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
    y1 = data[['lowestPrice']]
    y2 = data[['price_trend']]

    label_encoder = LabelEncoder()
    y2 = label_encoder.fit_transform(y2.values.ravel())
    joblib.dump(label_encoder, 'label_encoder.joblib')

    with tqdm(total=2, desc="Preprocessing steps") as pbar:
        X_preprocessed = preprocessor.fit_transform(X)
        pbar.update(1)

        joblib.dump(preprocessor, 'preprocessor.joblib')

        X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
            X_preprocessed, y1, y2, test_size=0.2, random_state=42)
        pbar.update(1)

    return X_train, X_test, y1_train, y1_test, y2_train, y2_test
