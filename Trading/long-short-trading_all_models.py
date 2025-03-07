import pandas as pd
import glob
import os

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np
import joblib
from matplotlib import dates as mdates

def long_short_accumulative_trading(df, enable_shorting=True, transaction_cost=1):
    initial_capital = 1000
    current_capital = initial_capital
    position = 0
    profit_loss = []
    trades = []
    stop_loss_percentage = 0.10  # 10% stop loss
    last_position_type = None  # To keep track of the last position type (long or short)

    # Sort the dataframe by tradeExecTime
    df = df.sort_values('tradeExecTime')

    # Convert tradeExecTime to datetime if it's not already
    df['tradeExecTime'] = pd.to_datetime(df['tradeExecTime'])



    # Group by trade_date
    for date, group in df.groupby(df['tradeExecTime'].dt.date):
        daily_profit_loss = 0
        entry_prices = []
        daily_position_value = 0
        daily_market_value = 0
        portfolio_value = 0
        prev_intraday_price = None
        trade_executed = False  # Flag to ensure only one trade per day
        neg_price = False
        neg_capital = False

        for i, row in group.iterrows():
            trades.append({
                'date': row['tradeExecTime'],
                'position': position,
                'price': row['intraday_price'],
                'type': 'BoD',
                'capital': current_capital,
                'predicted_intraday_price': row['predicted_intraday_price'],
                'predicted_last_intra_price': row['predicted_last_intra_price'],
                'daily_position_value': 0,
                'daily_market_value': daily_market_value,
                'portfolio_value': portfolio_value
            })

            # Check if current_capital is less than zero
            trade_executed = False
            if current_capital <= 0:
                neg_capital = True
                # Close all positions if any
                if position != 0:
                    profit = daily_market_value - daily_position_value
                    current_capital += daily_market_value
                    current_capital -= transaction_cost  # Record transaction cost
                    daily_profit_loss += profit - transaction_cost
                    trade_executed = True
                    portfolio_value = current_capital

                    trades.append({
                        'date': row['tradeExecTime'],
                        'position': 0,
                        'price': row['intraday_price'],
                        'type': 'close all (insufficient capital)',
                        'capital': current_capital,
                        'predicted_intraday_price': row['predicted_intraday_price'],
                        'predicted_last_intra_price': row['predicted_last_intra_price'],
                        'daily_position_value': 0,
                        'daily_market_value': daily_market_value,
                        'portfolio_value': portfolio_value
                    })
                    position = 0
                    entry_prices = []
                    daily_position_value = 0
                    daily_market_value = 0
                    last_position_type = None

            # Check for stop loss
            if position != 0 and prev_intraday_price is not None and not trade_executed:
                daily_market_value = position * row['intraday_price']
                potential_loss = prev_intraday_price * position
                if abs(potential_loss) > stop_loss_percentage * current_capital:
                    # Trigger stop loss
                    profit = daily_market_value - daily_position_value
                    current_capital += daily_market_value
                    current_capital -= transaction_cost  # Record transaction cost
                    daily_profit_loss += profit - transaction_cost
                    position = 0
                    entry_prices = []
                    daily_position_value = 0
                    daily_market_value = 0
                    portfolio_value = current_capital

                    trades.append({
                        'date': row['tradeExecTime'],
                        'position': 0,
                        'price': row['intraday_price'],
                        'type': 'stop loss',
                        'capital': current_capital,
                        'predicted_intraday_price': row['predicted_intraday_price'],
                        'predicted_last_intra_price': row['predicted_last_intra_price'],
                        'daily_position_value': daily_position_value,
                        'daily_market_value': daily_market_value,
                        'portfolio_value': portfolio_value,
                    })

                    last_position_type = None
                    trade_executed = True

            # Check if previous intraday price is less than or equal to zero
            if prev_intraday_price is not None and prev_intraday_price <= 0:
                neg_price = True
                # Close all positions
                if position != 0:
                    profit = daily_market_value - daily_position_value
                    current_capital += daily_market_value
                    current_capital -= transaction_cost  # Record transaction cost
                    daily_profit_loss += profit - transaction_cost
                    position = 0
                    entry_prices = []
                    daily_position_value = 0
                    daily_market_value = 0
                    portfolio_value = current_capital

                    trades.append({
                        'date': row['tradeExecTime'],
                        'position': 0,
                        'price': row['intraday_price'],
                        'type': 'close all (zero or negative price)',
                        'capital': current_capital,
                        'predicted_intraday_price': row['predicted_intraday_price'],
                        'predicted_last_intra_price': row['predicted_last_intra_price'],
                        'daily_position_value': 0,
                        'daily_market_value': daily_market_value,
                        'portfolio_value': portfolio_value
                    })
                    last_position_type = None

            if row['predicted_last_intra_price'] > 0 and row['predicted_intraday_price'] > 0 and not trade_executed and not neg_price and not neg_capital:
                price_difference = abs(row['predicted_last_intra_price'] - row['predicted_intraday_price'])
                if price_difference > transaction_cost:
                    if row['predicted_last_intra_price'] > row['predicted_intraday_price']:
                        # Go long or increase long position
                        if last_position_type == 'short':
                            # Close short position
                            profit = daily_position_value - (position * row['intraday_price'])
                            current_capital += (position * row['intraday_price'])
                            current_capital -= transaction_cost  # Record transaction cost
                            daily_profit_loss += profit - transaction_cost
                            position = 0
                            daily_position_value = 0
                            daily_market_value = 0
                            portfolio_value = current_capital
                            trades.append({
                                'date': row['tradeExecTime'],
                                'position': 0,
                                'price': row['intraday_price'],
                                'type': 'close short',
                                'capital': current_capital,
                                'predicted_intraday_price': row['predicted_intraday_price'],
                                'predicted_last_intra_price': row['predicted_last_intra_price'],
                                'daily_position_value': 0,
                                'daily_market_value': 0,
                                'portfolio_value': portfolio_value,
                            })
                            entry_prices = []

                        if (current_capital - row['intraday_price'] - transaction_cost) > 0:
                            new_position = position + 1
                            entry_price = row['intraday_price']
                            entry_prices.append(entry_price)
                            daily_position_value += entry_price
                            daily_market_value = new_position * entry_price
                            current_capital -= entry_price
                            current_capital -= transaction_cost  # Record transaction cost
                            portfolio_value = current_capital + daily_market_value
                            trades.append({
                                'date': row['tradeExecTime'],
                                'position': new_position,
                                'price': entry_price,
                                'type': 'long',
                                'capital': current_capital,
                                'predicted_intraday_price': row['predicted_intraday_price'],
                                'predicted_last_intra_price': row['predicted_last_intra_price'],
                                'daily_position_value': daily_position_value,
                                'daily_market_value': daily_market_value,
                                'portfolio_value': portfolio_value,
                            })
                            position = new_position
                            last_position_type = 'long'
                            trade_executed = True
                    elif row['predicted_last_intra_price'] < row['predicted_intraday_price'] and enable_shorting:
                        # Go short or increase short position
                        if last_position_type == 'long':
                            # Close long position
                            profit = (position * row['intraday_price']) - daily_position_value
                            current_capital += (position * row['intraday_price'])
                            current_capital -= transaction_cost  # Record transaction cost
                            daily_profit_loss += profit - transaction_cost
                            position = 0
                            daily_position_value = 0
                            daily_market_value = 0
                            entry_prices = []
                            portfolio_value = current_capital

                            trades.append({
                                'date': row['tradeExecTime'],
                                'position': 0,
                                'price': row['intraday_price'],
                                'type': 'close long',
                                'capital': current_capital,
                                'predicted_intraday_price': row['predicted_intraday_price'],
                                'predicted_last_intra_price': row['predicted_last_intra_price'],
                                'daily_position_value': 0,
                                'daily_market_value': 0,
                                'portfolio_value': portfolio_value
                            })
                        if current_capital > transaction_cost:
                            new_position = position - 1
                            entry_price = row['intraday_price']
                            entry_prices.append(entry_price)
                            daily_position_value -= entry_price
                            daily_market_value = new_position * entry_price
                            current_capital += entry_price
                            current_capital -= transaction_cost  # Record transaction cost
                            portfolio_value = current_capital + daily_market_value
                            trades.append({
                                'date': row['tradeExecTime'],
                                'position': new_position,
                                'price': entry_price,
                                'type': 'short',
                                'capital': current_capital,
                                'predicted_intraday_price': row['predicted_intraday_price'],
                                'predicted_last_intra_price': row['predicted_last_intra_price'],
                                'daily_position_value': daily_position_value,
                                'daily_market_value': daily_market_value,
                                'portfolio_value' : portfolio_value
                            })
                            position = new_position
                            last_position_type = 'short'
                            trade_executed = True
            if trade_executed == False:
                trades.append({
                    'date': row['tradeExecTime'],
                    'position': 0,
                    'price': row['intraday_price'],
                    'type': 'EoD',
                    'capital': current_capital,
                    'predicted_intraday_price': row['predicted_intraday_price'],
                    'predicted_last_intra_price': row['predicted_last_intra_price'],
                    'daily_position_value': 0,
                    'daily_market_value': daily_market_value,
                    'portfolio_value': portfolio_value
                })


            prev_intraday_price = row['intraday_price']

        # Materialize the position at the end of the day
        last_row = group.iloc[-1]
        if position != 0:
            # Close all positions
            profit = (last_row['last_intra_price'] * position) - daily_position_value
            current_capital += (last_row['last_intra_price'] * abs(position))
            current_capital -= transaction_cost  # Record transaction cost
            daily_profit_loss += profit - transaction_cost
            position = 0
            portfolio_value = current_capital
            trades.append({
                'date': last_row['tradeExecTime'],
                'position': 0,
                'price': last_row['last_intra_price'],
                'type': 'close all',
                'capital': current_capital,
                'predicted_intraday_price': last_row['predicted_intraday_price'],
                'predicted_last_intra_price': last_row['predicted_last_intra_price'],
                'daily_position_value': daily_position_value,
                'daily_market_value': position * last_row['last_intra_price'],
                'portfolio_value': portfolio_value,
            })

        profit_loss.append({'date': date, 'profit_loss': daily_profit_loss, 'capital': current_capital, 'portfolio_value': portfolio_value})
        position = 0  # Reset position for the next day
        last_position_type = None  # Reset last position type for the next day
        trade_executed = False
        neg_price = False

    return pd.DataFrame(profit_loss), pd.DataFrame(trades)


# Import the CSV file
base_path = r"C:\Users\chrsr\PycharmProjects\pythonProject\Models and outputs"
save_path = r"C:\Users\chrsr\PycharmProjects\pythonProject\Trading"

trading_run = input("Do you want to run trades? (yes/no): ").lower().strip() == 'yes'

if trading_run:
    all_trans_run = input("Run all trades between 0 and 1 transaction costs? (yes/no): ").lower().strip() == 'yes'
    if all_trans_run:
        transaction_costs = np.arange(0, 1.1, 0.1)
    else:
        transaction_costs = [float(input("Transaction cost level? (enter a number): "))]

    # Create a time range from 00:00 to 23:30 with 30-minute intervals
    time_range = pd.date_range(start='00:00', end='23:30', freq='30min').strftime('%H:%M')

    # Create the DataFrame with times and a running integer index starting from 1
    SP_overview_DF = pd.DataFrame({
        'SP': range(1, len(time_range) + 1),
        'Time': time_range
    })

    # Set 'Index' as a column instead of the DataFrame index
    SP_overview_DF.reset_index(drop=True, inplace=True)

    all_cumulative_profit_loss = {}
    all_trades = {}
    first = int(input("First SP to run? (enter a number): ")) - 1
    last = int(input("Last SP to run? (enter a number): "))

    for trans_costs in transaction_costs:
        # Initialize cumulative profit loss DataFrame and all trades DataFrame
        cumulative_profit_loss = pd.DataFrame()
        trades = pd.DataFrame()
        # Round transaction costs to the nearest 0.1
        trans_costs = round(trans_costs * 10) / 10
        for SP_value in SP_overview_DF['SP'][first:last]:
            # Get the specific start time for this SP
            specific_start = SP_overview_DF[SP_overview_DF['SP'] == SP_value]['Time'].values[0]
            specific_end = (pd.to_datetime(specific_start) + pd.Timedelta(minutes=30)).strftime('%H:%M')
            print(f'running model for SP {SP_value}, transaction cost {trans_costs}')
            name_string = specific_start.replace(':', '_')

            file_pattern = f"variables_model_intra_outputs_{name_string}.csv"
            input_path = str(os.path.join(base_path, file_pattern))

            df = pd.read_csv(input_path)

            # Check for duplicate tradeExecTime values
            duplicate_times = df[df.duplicated('tradeExecTime', keep=False)]

            if not duplicate_times.empty:
                print(f"Duplicate tradeExecTime values found. Removing duplicates...")
                # Keep only the first occurrence of each duplicate
                df = df.drop_duplicates(subset='tradeExecTime', keep='first')

            # Convert tradeExecTime to datetime format
            df['tradeExecTime'] = pd.to_datetime(df['tradeExecTime'])

            # Filter df to only include rows where "dataset" is equal to "test"
            df = df[df['dataset'] == 'trading']

            profit_loss_df, trades_df = long_short_accumulative_trading(df, enable_shorting=True, transaction_cost=trans_costs)

            # Add SP column to trades_df
            trades_df['SP'] = SP_value

            # Update cumulative profit loss
            if cumulative_profit_loss.empty:
                cumulative_profit_loss = profit_loss_df[['date', 'capital', 'portfolio_value']]
            else:
                temp_df = pd.merge(cumulative_profit_loss, profit_loss_df[['date', 'capital', 'portfolio_value']], on='date',
                                   how='outer', suffixes=('', '_new'))
                cumulative_profit_loss = pd.DataFrame({
                    'date': temp_df['date'],
                    'capital': temp_df['capital'].ffill().add(temp_df['capital_new'].ffill(), fill_value=0),
                    'portfolio_value': temp_df['portfolio_value'].ffill().add(temp_df['portfolio_value_new'].ffill(), fill_value=0)
                })

            # Append to all trades DataFrame
            trades = pd.concat([trades, trades_df], ignore_index=True)

        # Sort cumulative profit loss by date
        cumulative_profit_loss = cumulative_profit_loss.sort_values('date')

        # Calculate the daily difference from the previous period
        cumulative_profit_loss['daily_difference'] = cumulative_profit_loss['capital'].diff()

        # For the first row, calculate the difference from initial_capital
        # Multiply by the number of iterations (which is the number of SPs processed)
        num_iterations = len(SP_overview_DF['SP'][first:last])  # This is 1 in the current loop, but can be adjusted if needed
        initial_capital = 1000
        cumulative_profit_loss.loc[cumulative_profit_loss.index[0], 'daily_difference'] = (
            cumulative_profit_loss['capital'].iloc[0] - initial_capital
        ) * num_iterations

        # Sort all trades by date
        trades = trades.sort_values('date')

        # Calculate the development in capital rebased to 100
        initial_capital = cumulative_profit_loss['capital'].iloc[0]
        cumulative_profit_loss['capital_rebased'] = cumulative_profit_loss['capital'] / initial_capital * 100

        # Save cumulative profit loss and all trades to CSV files
        cumulative_profit_loss.to_csv(os.path.join(save_path, f'cumulative_profit_loss_cost_{trans_costs}.csv'), index=False)
        trades.to_csv(os.path.join(save_path, f'all_trades_cost_{trans_costs}.csv'), index=False)

        all_cumulative_profit_loss[trans_costs] = cumulative_profit_loss
        all_trades[trans_costs] = trades

    print("Cumulative profit loss and all trades data have been saved.")

# Read all cumulative profit loss files
if not trading_run:
    all_cumulative_profit_loss = {}
    for file in os.listdir(save_path):
        if file.startswith('cumulative_profit_loss_cost_') and file.endswith('.csv'):
            trans_costs = float(file.split('_')[-1].replace('.csv', ''))
            df = pd.read_csv(os.path.join(save_path, file))
            all_cumulative_profit_loss[trans_costs] = df

    all_trades = {}
    for file in os.listdir(save_path):
        if file.startswith('<all_trades_cost_>') and file.endswith('.csv'):
            trans_costs = float(file.split('_')[-1].replace('.csv', ''))
            df = pd.read_csv(os.path.join(save_path, file))
            all_trades[trans_costs] = df

import numpy as np

# Record the total number of trades and average trades per day across all transaction costs
total_trades = 0
total_trading_days = set()

# Input the specific transaction cost to analyze
# Example input: 0.1 (for 0.1 transaction cost)
specific_trans_cost = float(input("Enter the specific transaction cost to analyze: "))

if specific_trans_cost in all_trades:
    trades_df = all_trades[specific_trans_cost]
    # Ignore the first row in the CSV file
    trades_df = trades_df.iloc[1:]

    total_trades = len(trades_df[trades_df['type'] != None])
    trade_opportunities = len(trades_df)

    # Calculate total trading days
    total_trading_days = (pd.to_datetime(trades_df['date']).max() - pd.to_datetime(trades_df['date']).min()).days + 1

    # Group trades by SP
    grouped_trades = trades_df.groupby('SP')

    # Calculate percentage changes for each SP, only when stop loss or close conditions are met, based on capital
    sp_absolute_changes = []
    for _, sp_trades in grouped_trades:
        sp_trades = sp_trades.sort_values('date')
        previous_capital = None
        for _, trade in sp_trades.iterrows():
            current_capital = trade['portfolio_value']
            if previous_capital is not None and previous_capital != 0:
                absolute_change = (current_capital - previous_capital)
                sp_absolute_changes.append(absolute_change)
            previous_capital = current_capital

    # Aggregate percentage changes
    total_absolute_changes = sp_absolute_changes
    num_unique_trading_days = total_trading_days
    avg_trades_per_day = total_trades / num_unique_trading_days if num_unique_trading_days > 0 else 0

    avg_absolute_change = np.mean(total_absolute_changes) if total_absolute_changes else 0
    std_absolute_change = np.std(total_absolute_changes) if total_absolute_changes else 0

    # Calculate and display mean, max, and min percentage changes
    max_change = max(total_absolute_changes) if total_absolute_changes else 0
    min_change = min(total_absolute_changes) if total_absolute_changes else 0

    print(f"Total number of trades across all settlement periods: {total_trades} out {trade_opportunities} opportunities")
    print(f"Average trades per day across all settlement periods: {avg_trades_per_day:.2f}")
    print(f"Average change between closing trades: {avg_absolute_change:.2f}")
    print(f"Standard deviation of change between closing trades: {std_absolute_change:.2f}")

    # Plot distribution of total_percentage_changes
    plt.figure(figsize=(10, 6))
    plt.hist(total_absolute_changes, bins=50, edgecolor='black')
    plt.title('Distribution of Percentage Changes')
    plt.xlabel('Percentage Change')
    plt.ylabel('Frequency')
    plt.axvline(avg_absolute_change, color='r', linestyle='dashed', linewidth=2,
                label=f'Mean: {avg_absolute_change:.2f}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
else:
    print(f"No data available for transaction cost: {specific_trans_cost}")
    total_trades = 0
    total_trading_days = 0

# Create the plot

plt.figure(figsize=(12, 6))

for trans_costs, df in all_cumulative_profit_loss.items():
    print(df.head().to_string())
    plt.plot(df['date'], df['portfolio_value'], label=f'Transaction Cost: {trans_costs:.1f}')

# Customize the plot
plt.title('Development in value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio_value')

# Format x-axis to display dates nicely
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

# Rotate and align the tick labels so they look better
plt.gcf().autofmt_xdate()

# Set y-axis ticks
y_min, y_max = plt.ylim()
y_max_rounded = int(np.ceil(y_max / 10) * 10)
y_min_rounded = 0
y_range = y_max_rounded - y_min_rounded
tick_increment = y_range // 10  # 10 intervals for 11 ticks
tick_increment = max(100, int(np.ceil(tick_increment / 100) * 100))  # Round up to nearest 100, minimum 100
plt.yticks(range(y_min_rounded, y_max_rounded + tick_increment, tick_increment))

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Add a horizontal line at y=100 to show the initial capital level
plt.axhline(y=100, color='r', linestyle='--', label='Initial Capital')

# Add legend
plt.legend(loc='best')

# Show the plot
plt.tight_layout()
plt.show()
