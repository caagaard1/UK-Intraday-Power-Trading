import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np
import joblib

# Define the base path and file pattern
base_path = r"C:\Users\chrsr\PycharmProjects\pythonProject\intraday prices"
file_pattern = "intraday_1300_1330_*.csv"

# Create a time range from 00:00 to 23:30 with 30-minute intervals
time_range = pd.date_range(start='00:00', end='23:30', freq='30min').strftime('%H:%M')

# Create the DataFrame with times and a running integer index starting from 1
SP_overview_DF = pd.DataFrame({
    'SP': range(1, len(time_range) + 1),
    'Time': time_range
})

# Set 'Index' as a column instead of the DataFrame index
SP_overview_DF.reset_index(drop=True, inplace=True)

# Get a list of all matching CSV files
csv_files = glob.glob(os.path.join(base_path, file_pattern))

# Initialize an empty list to store dataframes
df_list = []

# Read each CSV file and append to the list
for file in csv_files:
    df = pd.read_csv(file, parse_dates=['tradeExecTime'])
    df_list.append(df)

# Concatenate all dataframes
combined_df = pd.concat(df_list, ignore_index=True)


# Convert tradeExecTime to datetime if it's not already
combined_df['tradeExecTime'] = pd.to_datetime(combined_df['tradeExecTime'], utc=True)
combined_df['tradeExecTime'] = combined_df['tradeExecTime'].dt.tz_localize(None)


# Loop through each SP in SP_overview_DF
for SP_value in SP_overview_DF['SP']:
    # Get the specific start time for this SP
    specific_start = SP_overview_DF[SP_overview_DF['SP'] == SP_value]['Time'].values[0]
    specific_end = (pd.to_datetime(specific_start) + pd.Timedelta(minutes=30)).strftime('%H:%M')

    print(f"SP {SP_value} - Combined shape: {combined_df.shape}")

    # Filter the combined dataframe
    filtered_df = combined_df[
        (combined_df['dlvryStart'].str.contains(specific_start)) &
        (combined_df['dlvryEnd'].str.contains(specific_end))
        ].copy()
    # Round tradeExecTime to the nearest minute

    filtered_df['dlvryStart'] = pd.to_datetime(filtered_df['dlvryStart'], utc=True)
    filtered_df['dlvryStart'] = filtered_df['dlvryStart'].dt.tz_localize(None)
    filtered_df['tradeExecTime'] = filtered_df['tradeExecTime'].dt.floor('min')
    filtered_df['tradeExecTime'] = pd.to_datetime(filtered_df['tradeExecTime'])

    print(f"SP {SP_value} - Filtered shape: {filtered_df.shape}")

    # Ensure dlvryStart is in a numeric format for mean calculation
    filtered_df['dlvryStart'] = filtered_df['dlvryStart'].astype(int) // 10 ** 9  # Convert to Unix timestamp

    # Group by the rounded tradeExecTime, including all columns
    grouped = filtered_df.groupby('tradeExecTime', as_index=False).agg({
        'volumeMW': 'sum',
        'price': 'mean',
        'dlvryStart': 'mean',
        # Include any other columns present in filtered_df
    })

    # Convert dlvryStart back to datetime
    filtered_df['dlvryStart'] = pd.to_datetime(filtered_df['dlvryStart'], unit='s')

    # Sort the dataframe by tradeExecTime
    filtered_df = filtered_df.sort_values('tradeExecTime')

    # Reset the index of the filtered dataframe
    filtered_df.reset_index(drop=True, inplace=True)

    print(f"SP {SP_value} - Final shape: {filtered_df.shape}")

    name_string = specific_start.replace(':', '_')

    # Define the output directory and filename
    output_directory = r'C:\Users\chrsr\PycharmProjects\pythonProject\intraday prices SP'
    output_filename = f'intra_file_{str(name_string)}.csv'

    # Combine the directory and filename to create the full path
    output_path = rf'{str(os.path.join(output_directory, output_filename))}'

    # Save the filtered dataframe as a CSV file
    filtered_df.to_csv(output_path, index=False)

    # Print confirmation message
    print(f"SP {SP_value} - {specific_start} to {specific_end}")
    print(f"File saved successfully: {output_path}")