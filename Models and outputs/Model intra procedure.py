import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np
import joblib

# Define the base path and file pattern
base_path = r"C:\Users\chrsr\PycharmProjects\pythonProject\intraday prices SP"
save_path = r"C:\Users\chrsr\PycharmProjects\pythonProject\Models and outputs"

# Create a time range from 00:00 to 23:30 with 30-minute intervals
time_range = pd.date_range(start='00:00', end='23:30', freq='30min').strftime('%H:%M')

# Create the DataFrame with times and a running integer index starting from 1
SP_overview_DF = pd.DataFrame({
    'SP': range(1, len(time_range) + 1),
    'Time': time_range
})

# Set 'Index' as a column instead of the DataFrame index
SP_overview_DF.reset_index(drop=True, inplace=True)

for SP_value in SP_overview_DF['SP']:
    # Get the specific start time for this SP
    specific_start = SP_overview_DF[SP_overview_DF['SP'] == SP_value]['Time'].values[0]
    specific_end = (pd.to_datetime(specific_start) + pd.Timedelta(minutes=30)).strftime('%H:%M')
    print(f'running model for {SP_value}')
    name_string = specific_start.replace(':', '_')
    file_pattern = f"intra_file_{name_string}.csv"

    input_path = str(os.path.join(base_path, file_pattern))

    filtered_df = pd.read_csv(input_path)

    print(filtered_df.shape, filtered_df.columns)

    dayahead_df = pd.read_csv(r'C:\Users\chrsr\PycharmProjects\pythonProject\dayahead.csv')
    dayahead_sp = dayahead_df[dayahead_df['settlement_period'] == SP_value].copy()
    dayahead_sp.reset_index(drop=True, inplace=True)

    # Extract date from tradeExecTime and dlvryStart
    filtered_df['tradeExecTime'] = pd.to_datetime(filtered_df['tradeExecTime'])
    filtered_df.dropna(subset='dlvryStart', inplace=True)
    filtered_df['delivery_date'] = pd.to_datetime(filtered_df['dlvryStart'], errors='coerce').dt.date
    # For rows where conversion failed, try to extract date from string
    mask = filtered_df['delivery_date'].isna()
    filtered_df.loc[mask, 'delivery_date'] = pd.to_datetime(
        filtered_df.loc[mask, 'dlvryStart'].str.extract(r'(\d{4}-\d{2}-\d{2})').squeeze(), errors='coerce').dt.date
    filtered_df['trade_date'] = filtered_df['tradeExecTime'].dt.date

    # Remove rows where delivery_date is still NaT after all attempts
    filtered_df.dropna(subset='delivery_date', inplace=True)

    # Extract only the delivery date and price_y from dayahead_sp
    dayahead_sp = dayahead_sp[['delivery_date', 'price']]

    # Convert delivery_date to datetime and extract date
    dayahead_sp['delivery_date'] = pd.to_datetime(dayahead_sp['delivery_date']).dt.date

    # Rename the price_y column to avoid confusion after merging
    dayahead_sp = dayahead_sp.rename(columns={'price_y': 'dayahead_price'})

    # Round tradeExecTime to the nearest minute
    filtered_df['tradeExecTime_minute'] = filtered_df['tradeExecTime'].dt.floor('min')

    # Group by the rounded tradeExecTime (minute), including all columns
    grouped = filtered_df.groupby('tradeExecTime_minute', as_index=False).agg({
        'volumeMW': 'sum',
        'price': 'mean',
        'delivery_date': 'first',
        'trade_date' : 'first'
        # Include any other columns present in filtered_df with appropriate aggregation
    })


    # Replace the original filtered_df with the grouped data
    filtered_df = grouped.rename(columns={'tradeExecTime_minute': 'tradeExecTime'})

    print(filtered_df.shape)

    # Merge based on the delivery date, keeping only matching rows from filtered_df
    merged_df = filtered_df.merge(dayahead_sp, on='delivery_date', how='left')

    # Rename price_y to dayahead_price
    merged_df = merged_df.rename(columns={'price_y': 'dayahead_price', 'price_x': 'intraday_price'})

    wind_factor_df = pd.read_csv(r'C:\Users\chrsr\PycharmProjects\pythonProject\weather\wind_factor.csv')
    wind_factor_df = wind_factor_df.iloc[0:]
    wind_factor_df['datetime'] = pd.to_datetime(wind_factor_df['datetime'])

    # Extract date, year, and hour from tradeExecTime and datetime
    merged_df['date_2'] = merged_df['tradeExecTime'].dt.date
    merged_df['year_2'] = merged_df['tradeExecTime'].dt.year
    merged_df['hour_2'] = merged_df['tradeExecTime'].dt.hour

    wind_factor_df['date_2'] = wind_factor_df['datetime'].dt.date
    wind_factor_df['year_2'] = wind_factor_df['datetime'].dt.year
    wind_factor_df['hour_2'] = wind_factor_df['datetime'].dt.hour

    # Merge wind_factor_df into merged_df
    merged_df = merged_df.merge(wind_factor_df, on=['date_2', 'year_2', 'hour_2'], how='left')

    # Drop the temporary columns used for merging
    merged_df = merged_df.drop(columns=['date_2', 'year_2', 'hour_2'])

    # Drop the 'datetime' column from wind_factor_df if it's not needed
    merged_df = merged_df.drop(columns=['datetime'], errors='ignore')

    # Update filtered_df with the merged data
    filtered_df = merged_df

    # Sort filtered_df by TradeExecTime
    filtered_df = filtered_df.sort_values('tradeExecTime', ascending=False)

    # Find the last intraday price for each date based on the last tradeExecTime
    last_prices = filtered_df.groupby('delivery_date').apply(
        lambda x: pd.Series({
            'delivery_date': x.name,
            'last_intra_price': x.loc[x['tradeExecTime'].idxmax(), 'intraday_price']
        })
    ).reset_index(drop=True)

    # Ensure the output is a DataFrame with two columns
    last_prices = last_prices[['delivery_date', 'last_intra_price']]

    # Merge the last prices back to the original dataframe
    filtered_df = pd.merge(filtered_df, last_prices, on='delivery_date', how='left')

    # Reorder columns to place new columns after 'price'
    cols = filtered_df.columns.tolist()
    price_index = cols.index('intraday_price')
    new_cols = cols[:price_index + 1] + ['last_intra_price'] + [col for col in cols[price_index + 1:]
                                                                                   if
                                                                                   col not in ['last_intra_price']]
    filtered_df = filtered_df[new_cols]

    #rename dataframe
    intra_file_df = filtered_df.copy()

    # Select the required columns
    selected_columns = ['tradeExecTime', 'intraday_price', 'dayahead_price', 'trade_date', 'last_intra_price', 'delivery_date', 'Wind factor H-1']
    intra_file_df = intra_file_df[selected_columns]

    # Create lagged columns for intraday_price
    intra_file_df['intraday_price_lag_1'] = intra_file_df['intraday_price'].shift(1)
    intra_file_df['intraday_price_lag_5'] = intra_file_df['intraday_price'].shift(5)
    intra_file_df['intraday_price_lag_10'] = intra_file_df['intraday_price'].shift(10)

    # Create a new column for percentage change from lag 10 to lag 1
    intra_file_df['pct_change_lag10_to_lag1'] = pd.to_numeric(
        (intra_file_df['intraday_price_lag_1'] - intra_file_df['intraday_price_lag_10']) / intra_file_df['intraday_price_lag_10'],
        errors='coerce'
    )

    # Sort the dataframe by tradeExecTime in ascending order
    intra_file_df = intra_file_df.sort_values('tradeExecTime', ascending=True)

    # Calculate the price difference between day-ahead price (dayahead_price) and intraday price (intraday_price)
    intra_file_df['price_diff'] = intra_file_df['dayahead_price'] - intra_file_df['intraday_price']

    # Calculate the average intraday_price for the range 10 to 70 minutes before each row's tradeExecTime
    intra_file_df['last_hour_avg_price'] = intra_file_df.apply(
        lambda row: intra_file_df[
            (intra_file_df['tradeExecTime'] <= row['tradeExecTime'] - pd.Timedelta(minutes=10)) &
            (intra_file_df['tradeExecTime'] > row['tradeExecTime'] - pd.Timedelta(minutes=70))
            ]['intraday_price'].mean(),
        axis=1
    )
    # Extract day, hour, and minute from tradeExecTime
    intra_file_df['day'] = intra_file_df['tradeExecTime'].dt.day
    intra_file_df['hour'] = intra_file_df['tradeExecTime'].dt.hour
    intra_file_df['minute'] = intra_file_df['tradeExecTime'].dt.minute

    # Create counters that start from 0 and run throughout all the data
    intra_file_df['minute_counter'] = (intra_file_df['tradeExecTime'] - intra_file_df[
        'tradeExecTime'].min()).dt.total_seconds() / 60
    intra_file_df['hour_counter'] = intra_file_df['minute_counter'] // 60
    intra_file_df['day_counter'] = intra_file_df['hour_counter'] // 24

    # Convert counters to integers
    intra_file_df['minute_counter'] = intra_file_df['minute_counter'].astype(int)
    intra_file_df['hour_counter'] = intra_file_df['hour_counter'].astype(int)
    intra_file_df['day_counter'] = intra_file_df['day_counter'].astype(int)

    # Group by day_counter and calculate daily volatility
    daily_volatility = intra_file_df.groupby('day_counter')['price_diff'].std().reset_index()
    daily_volatility.columns = ['day_counter', 'volatility']

    # Calculate the average volatility for each day count
    avg_volatility_by_day = daily_volatility.groupby('day_counter')['volatility'].mean().reset_index()
    avg_volatility_by_day.columns = ['day_counter', 'avg_volatility']

    # Merge avg_volatility_by_day with intra_file_df based on day_counter
    intra_file_df = pd.merge(intra_file_df, avg_volatility_by_day, on='day_counter', how='left')

    # Rename the 'avg_volatility' column to make it more descriptive
    intra_file_df = intra_file_df.rename(columns={'avg_volatility': 'avg_volatility_by_day'})

    # Create a new column with the average volatility for the previous day
    intra_file_df['avg_volatility_previous_day'] = intra_file_df['day_counter'].map(
        intra_file_df.groupby('day_counter')['avg_volatility_by_day'].first().shift(1)
    )

    # Calculate the volatility of price_diff from the beginning of the day until 10 minutes before the current time
    intra_file_df['volatility_until_10min_before'] = intra_file_df.groupby('day_counter').apply(
        lambda group: group.apply(
            lambda row: group[
                (group['minute_counter'] < row['minute_counter'] - 10) &
                (group['minute_counter'] >= group['minute_counter'].min())
                ]['price_diff'].std(),
            axis=1
        )
    ).reset_index(level=0, drop=True)

    # Replace NaN values with 0 for the first 10 minutes of each day
    intra_file_df['volatility_until_10min_before'] = intra_file_df['volatility_until_10min_before'].fillna(0)

    intra_file_df = intra_file_df.dropna()
    intra_file_df = intra_file_df[~intra_file_df.isin([np.inf, -np.inf]).any(axis=1)]

    # Create a reference time of 13:00 for each day
    intra_file_df['reference_time'] = intra_file_df['tradeExecTime'].dt.normalize() + pd.Timedelta(hours=SP_value/2)

    # Calculate the time difference in minutes
    intra_file_df[f'minutes_from_{specific_start}'] = (intra_file_df['tradeExecTime'] - intra_file_df[
        'reference_time']).dt.total_seconds() / 60

    # Round to nearest minute and convert to integer
    intra_file_df[f'minutes_from_{specific_start}'] = intra_file_df[f'minutes_from_{specific_start}'].round().astype(int)

    # Set up training and test data sets
    individual_features = ['dayahead_price', 'last_hour_avg_price',
                           'intraday_price_lag_5', 'Wind factor H-1',
                           'intraday_price_lag_10', 'pct_change_lag10_to_lag1', 'avg_volatility_previous_day',
                           'volatility_until_10min_before']

    print(intra_file_df.head().to_string())
    print(intra_file_df.shape)

    X = intra_file_df[individual_features]
    # Check for errors, NaN, or infinite values
    has_nan = X.isna().any().any()
    has_inf = np.isinf(X).any().any()

    if has_nan or has_inf:
        print("\nChecking for errors, NaN, or infinite values:")
        print(f"NaN values present in {specific_start}: {has_nan}")
        print(f"Infinite values present in {specific_start}: {has_inf}")

    y = intra_file_df['intraday_price']

    # Split the data into training (60%) and testing (40%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    X_test, X_trading, y_test, y_trading = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

    # Create and train the XGBRegressor
    xgb_model_intra = XGBRegressor(n_estimators=500, early_stopping_rounds=10, learning_rate=0.2)
    xgb_model_intra.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Make predictions on the test set
    y_pred = xgb_model_intra.predict(X_test)

    # Save the trained model
    output_path = str(os.path.join(save_path, f'xgb_model_intra_{name_string}.joblib'))

    joblib.dump(xgb_model_intra, output_path)

    # Create a new DataFrame with intra_file_df, predicted values, and errors for both train and test sets
    output_df = intra_file_df.copy()

    # Predict values for both train and test sets
    y_pred_train = xgb_model_intra.predict(X_train)
    y_pred_test = y_pred  # y_pred is already calculated for the test set
    y_pred_trading = xgb_model_intra.predict(X_trading)

    # Combine predictions
    all_predictions = np.concatenate([y_pred_train, y_pred_test, y_pred_trading])
    output_df['predicted_intraday_price'] = all_predictions

    # Calculate errors for both train and test sets
    errors_train = y_train - y_pred_train
    errors_test = y_test - y_pred_test
    errors_trading = y_trading - y_pred_trading

    # Combine errors
    all_errors = np.concatenate([errors_train, errors_test, errors_trading])
    output_df['model_error'] = all_errors

    # Add a categorical variable for train/test/trading split
    train_index = len(X_train)
    test_index = train_index + len(X_test)
    output_df['dataset'] = ['train' if i < train_index else 'test' if i < test_index else 'trading' for i in
                            range(len(output_df))]

    # Save the DataFrame as a CSV file
    output_path = str(os.path.join(save_path, f'variables_model_intra_outputs_{name_string}.csv'))
    output_df.to_csv(output_path, index=False)

    print(f"DataFrame saved as 'variables_model_intra_outputs_{specific_start}.csv'")