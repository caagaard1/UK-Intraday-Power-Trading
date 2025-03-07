import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np
import joblib

# Define the base path and file pattern
base_path = r"C:\Users\chrsr\PycharmProjects\pythonProject\Models and outputs"
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
    file_pattern = f"variables_model_intra_outputs_{name_string}.csv"

    input_path = str(os.path.join(base_path, file_pattern))

    last_file_df = pd.read_csv(input_path)

    # Set up training and test data sets
    individual_features = ['dayahead_price', 'last_hour_avg_price',
                           'intraday_price_lag_5', 'Wind factor H-1',
                           'intraday_price_lag_10', 'pct_change_lag10_to_lag1', 'avg_volatility_previous_day',
                           'volatility_until_10min_before']

    X = last_file_df[individual_features]
    # Check for errors, NaN, or infinite values
    has_nan = X.isna().any().any()
    has_inf = np.isinf(X).any().any()

    if has_nan or has_inf:
        print("\nChecking for errors, NaN, or infinite values:")
        print(f"NaN values present in {specific_start}: {has_nan}")
        print(f"Infinite values present in {specific_start}: {has_inf}")

    y = last_file_df['last_intra_price']

    # Split the data into training (60%) and testing (40%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    X_test, X_trading, y_test, y_trading = train_test_split(X_test, y_test, test_size = 0.5, random_state=0)

    # Create and train the XGBRegressor
    xgb_model_last = XGBRegressor(n_estimators=500, early_stopping_rounds=10, learning_rate=0.2)
    xgb_model_last.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Make predictions on the test set
    y_pred = xgb_model_last.predict(X_test)

    # Save the trained model
    output_path = str(os.path.join(save_path, f'xgb_model_last_{name_string}.joblib'))

    joblib.dump(xgb_model_last, output_path)

    # Create a new DataFrame with last_file_df, predicted values, and errors for both train and test sets
    output_df = last_file_df.copy()

    # Predict values for both train and test sets
    y_pred_train = xgb_model_last.predict(X_train)
    y_pred_test = y_pred  # y_pred is already calculated for the test set
    y_pred_trading = xgb_model_last.predict(X_trading)

    # Combine predictions
    all_predictions = np.concatenate([y_pred_train, y_pred_test, y_pred_trading])
    output_df['predicted_last_intra_price'] = all_predictions

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

    print(f"DataFrame saved as 'variables_model_outputs_{specific_start}.csv'")