import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO
import csv
import requests
import json
import urllib3
import os

# Input to determine whether to update data from API
update_data = int(input("Enter 1 to update data from API, 0 to use stored data: "))

csv_filename = 'dayahead.csv'

if update_data == 1 or not os.path.exists(csv_filename):
    # Make the API request
    url = 'https://api.modoenergy.com/pub/v1/gb/epex/day-ahead/hh'
    params = {
        'date_from': '2023-09-22',
        'date_to': '2024-09-22',
        'limit': '10000',  # Set a reasonable limit per request
        'offset': '0',
        'timezone': 'Europe/London',
    }
    headers = {
        'X-Token': 'e7065406cfdf8b2f903427c416892d3a96cfc4d696696f5b1ec2ddca6d1f',
        'accept': 'application/json'
    }

    all_results = []

    iterations_i = 0

    while True:
        iterations_i += 1
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        results = data['results']
        all_results.extend(results)
        print(f'Iteration number: {iterations_i}')

        if len(results) < int(params['limit']):
            break

        params['offset'] = str(int(params['offset']) + len(results))

    df = pd.DataFrame(all_results)

    # Keep only the specified columns
    df = df[['start_time', 'price', 'settlement_period', 'delivery_date', 'volume']]

    # Set tradeExecTime as the index and sort in ascending order
    df['start_time'] = pd.to_datetime(df['start_time'])
    df = df.set_index('start_time').sort_index(ascending=True)

    # Store the dataframe as a CSV file
    df.to_csv(csv_filename)
    print("Data updated from API and saved locally.")
else:
    # If update_data is 0 and the file exists, read from the CSV
    df = pd.read_csv(csv_filename, parse_dates=['start_time'])
    df.set_index('start_time', inplace=True)
    print("Data loaded from local storage.")

# Now df is ready to use, either freshly fetched and saved, or loaded from the CSV

# Display overview of saved dataframe

print(df.shape)


#settlement period 27