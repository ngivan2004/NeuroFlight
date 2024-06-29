import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def plot_category_distribution(data, column_name):
    counter = Counter(data[column_name])
    categories = list(counter.keys())
    counts = list(counter.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=categories, y=counts, palette="viridis")
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()


def check_imbalance(data, categorical_features, target_features):
    for feature in categorical_features:
        print(f"Checking imbalance for input feature: {feature}")
        plot_category_distribution(data, feature)

    for feature in target_features:
        print(f"Checking imbalance for target feature: {feature}")
        plot_category_distribution(data, feature)


def main():
    file_path = './data/cleanest_iten.csv'  # Replace with your actual file path
    data = load_data(file_path)

    categorical_features = ['startingAirport',
                            'destinationAirport', 'segmentsAirlineCode']
    target_features = ['lowestPrice', 'lowestPriceDayLeft']

    check_imbalance(data, categorical_features, target_features)


if __name__ == "__main__":
    main()
