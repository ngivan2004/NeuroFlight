def get_total_rows(file_path):
    with open(file_path) as f:
        for i, l in enumerate(f):
            pass
    return i


file_path = './data/processed_flights.csv'
total_rows = get_total_rows(file_path)
print(total_rows)
