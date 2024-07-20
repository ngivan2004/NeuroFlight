import requests


url = 'https://1zkf8hrdfk.execute-api.us-east-1.amazonaws.com/default/airline-api'
json_data = {
    "startingAirport": "ATL",
    "destinationAirport": "LAX",
    "segmentsAirlineCode": "DL",
    "travelDurationHours": 4,
    "travelDurationMinutes": 30,
    "totalFare": 300,
    "flightDate": "20/07/2024"
}



response = requests.get(url, json=json_data, headers={
    'Content-Type': 'application/json'})

if response.status_code == 200:
    print('Success:', response.json())
else:
    print('Failed:', response.status_code, response.text)
