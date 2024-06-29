import pandas as pd
import numpy as np
from tqdm import tqdm


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def cap_frequency(data, column, target_value):
    # Calculate the mean frequency of all OTHER values
    value_counts = data[column].value_counts()
    other_values_mean_frequency = value_counts[value_counts.index != target_value].mean(
    )
    mean_frequency = int(other_values_mean_frequency)
    print(
        f"Mean frequency of {column} excluding {target_value}: {mean_frequency}")

    target_indices = data[data[column] == target_value].index

    if len(target_indices) > mean_frequency:
        # Randomly select indices to keep
        keep_indices = np.random.choice(
            target_indices, size=mean_frequency, replace=False)
        # Drop excess indices
        drop_indices = set(target_indices) - set(keep_indices)
        data = data.drop(drop_indices)

    return data


def save_data(data, new_file_path):
    data.to_csv(new_file_path, index=False)
    print(f"Data saved to {new_file_path}")


def main():
    file_path = './data/cleaner_iten.csv'

    new_file_path = './data/cleanest_iten.csv'

    data = load_data(file_path)

    capped_data = cap_frequency(data, 'lowestPriceDayLeft', 60)

    save_data(capped_data, new_file_path)


if __name__ == "__main__":
    main()
