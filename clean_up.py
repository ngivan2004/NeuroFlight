import pandas as pd
from tqdm import tqdm


chunk_size = 100000
chunks = []
min_price_info = {}
total_rows = sum(1 for _ in open('./data/itineraries.csv', 'r')
                 ) - 1  # minus 1 for header
progress_bar = tqdm(total=total_rows, desc="Processing Chunks")

for chunk in pd.read_csv('./data/itineraries.csv', chunksize=chunk_size):
    chunk = chunk[chunk['isNonStop']]  # Filter non-stop flights
    chunk = chunk[~chunk['isBasicEconomy']]  # ..
    chunk = chunk[~chunk['isRefundable']]  # ..
    chunk = chunk[chunk['segmentsCabinCode'] == 'coach']  # ..
    chunk['searchDate'] = pd.to_datetime(chunk['searchDate'])
    chunk['flightDate'] = pd.to_datetime(chunk['flightDate'])
    chunk['daysUntilFlight'] = (
        chunk['flightDate'] - chunk['searchDate']).dt.days
    chunks.append(chunk)

    # Update min_price_info with the minimum price and corresponding date for each legID
    for _, row in chunk.iterrows():
        leg_id = row['legId']
        total_fare = row['totalFare']
        search_date = row['searchDate']
        if (leg_id not in min_price_info) or (total_fare < min_price_info[leg_id]['lowestPrice']):
            min_price_info[leg_id] = {
                'lowestPrice': total_fare,
                'lowestPriceDate': search_date
            }

    progress_bar.update(chunk.shape[0])

progress_bar.close()
df = pd.concat(chunks)

# Convert min_price_info to DataFrame for merging
min_price_df = pd.DataFrame.from_dict(
    min_price_info, orient='index').reset_index()
min_price_df.rename(columns={'index': 'legId'}, inplace=True)

# Step 2: Merge the min_price_df back into the main DataFrame
df = df.merge(min_price_df, on='legId', how='left')

# Step 3: Calculate lowestPriceDayLeft
df['lowestPriceDayLeft'] = (df['flightDate'] - df['lowestPriceDate']).dt.days

# Step 4: Drop the specified columns
columns_to_drop = [
    'isNonStop', 'seatsRemaining', 'totalTravelDistance', 'segmentsDepartureTimeEpochSeconds',
    'segmentsDepartureTimeRaw', 'segmentsArrivalTimeEpochSeconds', 'segmentsArrivalTimeRaw',
    'segmentsArrivalAirportCode', 'segmentsDepartureAirportCode', 'segmentsEquipmentDescription',
    'segmentsDurationInSeconds', 'segmentsDistance', 'segmentsCabinCode', 'fareBasisCode',
    'elapsedDays', 'segmentsAirlineName', 'isBasicEconomy', 'isRefundable'
]
df.drop(columns=columns_to_drop, inplace=True)

# Step 5: Remove the year from date columns
df['searchDate'] = df['searchDate'].dt.strftime('%m-%d')
df['flightDate'] = df['flightDate'].dt.strftime('%m-%d')
df['lowestPriceDate'] = df['lowestPriceDate'].dt.strftime('%m-%d')

# Step 6: Save the modified DataFrame
df.to_csv('./data/clean_iten.csv', index=False)
