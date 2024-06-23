import pandas as pd
import os
import sys
from collections import defaultdict


file_path = 'itineraries.csv'


legId_rows = defaultdict(list)
duplicate_entries = []


chunk_size = 100000


total_rows = 82000000
chunks_read = 0


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()


for chunk in pd.read_csv(file_path, chunksize=chunk_size):

    chunks_read += chunk_size
    print_progress_bar(chunks_read, total_rows,
                       prefix='Progress:', suffix='Complete', length=50)

    for index, row in chunk.iterrows():
        legId = row['legId']
        row_tuple = tuple(row)  # Convert row to a tuple to check for equality
        legId_rows[legId].append(row_tuple)

        # Check if we have found a duplicate with the same content
        if len(legId_rows[legId]) == 2:
            if legId_rows[legId][0] == legId_rows[legId][1]:
                duplicate_entries.append(row)
                if len(duplicate_entries) == 5:
                    break
    if len(duplicate_entries) == 5:
        break


for entry in duplicate_entries:
    print(entry)

# Ensure progress bar reaches 100% in case of early break
print_progress_bar(total_rows, total_rows, prefix='Progress:',
                   suffix='Complete', length=50)
