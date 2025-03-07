import requests
import pandas as pd
from datetime import datetime, timedelta


def fetch_elexon_physical_pn(date_from, date_to, bm_unit):
    base_url = "https://api.modoenergy.com/pub/v1/gb/elexon/physical/pn"
    headers = {
        'X-Token': '490a07f38c83de8ce889353bccbd0a8612d87ae6b49e1178bd31037bd61c',
        'accept': 'application/json'
    }
    params = {
        'date_from': date_from,
        'date_to': date_to,
        'limit': '10000',
        'timezone': 'Europe/London',
        'bm_unit': bm_unit,
        'offset' : '0'
    }

    all_results = []

    while True:
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        results = data.get('results', [])

        if not results:
            break

        all_results.extend(results)

        if len(results) < int(params['limit']):
            break

        # Update the date_from parameter for the next request
        params['offset'] = str(int(params['offset']) + len(results))

    return pd.DataFrame(all_results)


# Read the wind farms data
wind_farms_df = pd.read_excel(r"C:\Users\chrsr\PycharmProjects\pythonProject\.venv\wind_farms_uk.xlsx")

# Extract BM units from the wind farms data
bm_units = wind_farms_df['BMU_ID'].tolist()

# Example usage
date_from = "2023-09-22T00:00:00"
date_to = "2024-09-22T00:00:00"

# Initialize an empty list to store dataframes for each BM unit
dfs = []
A_num = 0

# Fetch data for each BM unit
for bm_unit in bm_units:
    A_num += 1
    bm_unit = str(bm_unit)
    print(f'asset number {A_num} - {bm_unit}')
    df = fetch_elexon_physical_pn(date_from, date_to, bm_unit)
    if not df.empty:
        df['levelAvg'] = (df['levelFrom'] + df['levelTo']) / 2
        dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Group by timeFrom and bmUnit, and aggregate levelAvg
grouped_df = combined_df.groupby(['timeFrom', 'bmUnit'])['levelAvg'].mean().reset_index()

# Pivot the table to create columns for each BM unit
df_pivot = grouped_df.pivot(index='timeFrom', columns='bmUnit', values='levelAvg')

# Sort the dataframe by timeFrom
df_pivot.sort_index(inplace=True)

# Reset the index to make timeFrom a column
df_pivot.reset_index(inplace=True)

# Remove all rows where a NaN value is displayed
df_pivot = df_pivot.dropna()

print(f"Processed data for {len(bm_units)} BM units.")
print(df_pivot.head())

# Save the result to a CSV file
df_pivot.to_csv('PN wind assets.csv', index=False)
print("Data saved to 'PN wind assets.csv'")