import pandas as pd
import itertools
from tqdm import tqdm

# Airports present in the dataset
airports = ['ATL', 'DFW', 'DEN', 'ORD', 'LAX', 'CLT', 'MIA',
            'JFK', 'EWR', 'SFO', 'DTW', 'BOS', 'PHL', 'LGA', 'IAD', 'OAK']

# Generate all possible routes
all_possible_routes = set(itertools.product(airports, airports))

# Remove routes where starting and destination are the same
all_possible_routes = {(start, end)
                       for start, end in all_possible_routes if start != end}

# Function to get the total number of rows in the CSV file to get a progress bar working lol


def get_total_rows(file_path):
    with open(file_path) as f:
        for i, l in enumerate(f):
            pass
    return i


# Get the total number of rows in the CSV file
file_path = './data/processed_flights.csv'
total_rows = get_total_rows(file_path)
chunksize = 100000
total_chunks = total_rows // chunksize + 1

# get all routes


def read_routes(file_path, chunksize=100000, total_chunks=total_chunks):
    routes = set()
    for chunk in tqdm(pd.read_csv(file_path, usecols=['startingAirport', 'destinationAirport'], chunksize=chunksize), total=total_chunks):
        chunk_routes = set(
            zip(chunk['startingAirport'], chunk['destinationAirport']))
        routes.update(chunk_routes)
    return routes


dataset_routes = read_routes(file_path)


missing_routes = all_possible_routes - dataset_routes


missing_routes_list = list(missing_routes)
print(len(all_possible_routes))
print(len(dataset_routes))
print(f'Total missing routes: {len(missing_routes_list)}')
print(missing_routes_list)
