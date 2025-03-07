import requests
import pandas as pd
from io import StringIO


def fetch_weather_data(latitude, longitude, start_date, end_date, forecast_basis_day=1):
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
    api_key = "R2BYX5VAUH8YSF6VH3VNKWPLZ"  # Replace with your actual API key if different

    # Convert start_date and end_date to datetime objects
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Initialize an empty DataFrame to store all data
    all_data = pd.DataFrame()

    # Set up base parameters
    params = {
        "unitGroup": "metric",
        "include": "hours",
        "key": api_key,
        "contentType": "csv",
    }

    # Add forecastBasisDay parameter only if it's not zero
    if forecast_basis_day != 0:
        params["forecastBasisDay"] = forecast_basis_day

    # Paginate through the data in 10-day chunks
    current_start = start
    while current_start <= end:
        current_end = min(current_start + pd.Timedelta(days=9), end)

        # Construct the URL for the current segment
        location = f"{latitude}%2C{longitude}"
        url = f"{base_url}{location}/{current_start.strftime('%Y-%m-%d')}/{current_end.strftime('%Y-%m-%d')}"

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Convert CSV content to DataFrame and append to all_data
            segment_df = pd.read_csv(StringIO(response.text))
            all_data = pd.concat([all_data, segment_df], ignore_index=True)

        except requests.RequestException as e:
            print(f"Error fetching weather data for segment {current_start} to {current_end}: {e}")

        # Move to the next segment
        current_start = current_end + pd.Timedelta(days=1)

    return all_data if not all_data.empty else None

start_date = "2023-09-22"
end_date = "2024-09-22"
forecast_basis_days = [0, 1]

if __name__ == "__main__":
    # Read the wind farms data
    wind_farms_df = pd.read_excel(r"C:\Users\chrsr\PycharmProjects\pythonProject\.venv\wind_farms_uk.xlsx")

    # Initialize an empty DataFrame to store all weather data
    all_weather_data = pd.DataFrame()

    for forecast_basis_day in forecast_basis_days:
        for index, row in wind_farms_df.iterrows():
            latitude, longitude = row['Coordinates'].split(',')
            latitude = float(latitude.strip())
            longitude = float(longitude.strip())
            wind_farm_name = row['Wind Farm']

            weather_df = fetch_weather_data(latitude, longitude, start_date, end_date, forecast_basis_day)

            if weather_df is not None:
                # Extract datetime and wind speed
                weather_data = weather_df[['datetime', 'windspeed']]
                weather_data.set_index('datetime', inplace=True)
                # Add the new column to all_weather_data
                if all_weather_data.empty:
                    all_weather_data = weather_data
                    new_column_name_1 = f"{wind_farm_name} t-{forecast_basis_day} - {latitude}, {longitude}"
                    all_weather_data.update(weather_data.rename(columns={'windspeed': new_column_name_1}))
                else:
                    # Add a new column to all_weather_data for the current wind farm
                    new_column_name = f"{wind_farm_name} t-{forecast_basis_day} /n {latitude}, {longitude}"
                    all_weather_data[new_column_name] = pd.Series(dtype='float64')

                    # Align and add data from weather_data to all_weather_data
                    all_weather_data.update(weather_data.rename(columns={'windspeed': new_column_name}))
            else:
                print(f"Failed to fetch weather data for {wind_farm_name} with forecast basis day {forecast_basis_day}")

    if not all_weather_data.empty:
        print("Weather data fetched successfully and stored as a DataFrame.")
        print(all_weather_data.head())  # Display the first few rows of the DataFrame
        print(f"DataFrame shape: {all_weather_data.shape}")
        print(f"DataFrame columns: {all_weather_data.columns.tolist()}")

        # Optionally, you can save the DataFrame to a CSV file
        # all_weather_data.to_csv("all_wind_farms_weather_data.csv")
    else:
        print("Failed to fetch weather data for all wind farms.")

    # Save the DataFrame to a CSV file
    csv_filename = "all_wind_farms_weather_data.csv"
    all_weather_data.to_csv(csv_filename)
    print(f"Weather data saved to {csv_filename}")
