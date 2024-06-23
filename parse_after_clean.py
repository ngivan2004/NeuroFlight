import pandas as pd
from tqdm import tqdm
import re
from datetime import datetime


def parse_duration(duration):
    pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?')
    match = pattern.match(duration)
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    return hours * 60 + minutes


def parse_flight_date(date_str):
    year = 2022  # As specified
    date = datetime.strptime(f"{year}-{date_str}", "%Y-%m-%d")
    return pd.Series([date.weekday(), date.day, date.month], index=['flightDayOfWeek', 'flightDayOfMonth', 'flightMonth'])


def process_chunk(chunk):
    chunk['travelDuration'] = chunk['travelDuration'].apply(parse_duration)
    date_components = chunk['flightDate'].apply(parse_flight_date)
    chunk = pd.concat([chunk, date_components], axis=1)
    return chunk


def process_large_csv(file_path, output_path, chunk_size=100000):
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    num_chunks = sum(1 for _ in open(file_path)) // chunk_size

    with tqdm(total=num_chunks, desc="Processing CSV") as pbar:
        for chunk in chunks:
            processed_chunk = process_chunk(chunk)
            processed_chunk.to_csv(
                output_path, mode='a', index=False, header=not pd.io.common.file_exists(output_path))
            pbar.update(1)


# File paths
input_file_path = './data/clean_iten.csv'
output_file_path = './data/cleaner_iten.csv'

# Process the large CSV
process_large_csv(input_file_path, output_file_path)
