import pandas as pd
import glob
import os

from matplotlib import pyplot as plt
from openpyxl.utils.dataframe import dataframe_to_rows
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np
import joblib
from matplotlib import dates as mdates


def long_short_trading_strategy(df, initial_capital=1000, transaction_cost=1, avg_periods = 1):
    # Ensure the dataframe is sorted by date and time
    df = df.sort_values('tradeExecTime')

    # Initialize variables
    current_capital = initial_capital
    position = 0
    acquisition_value = 0
    avg_acquisition_value = 0
    trades = []
    daily_aggregates = []
    aggregate_trade_pnl = 0
    previous_price = 0
    just_closed = False
    stop_loss_percentage = 0.2

    # Group by date
    for date, group in df.groupby(pd.to_datetime(df['tradeExecTime']).dt.date):
        start_capital = current_capital
        num_trades = 0

        for i, (_, row) in enumerate(group.iterrows()):
            # Calculate the difference between predicted prices  # You can adjust this value as needed
            last_x_rows = group.iloc[max(0, i - avg_periods + 1):i + 1]
            price_difference = (last_x_rows['predicted_last_intra_price'] - last_x_rows['predicted_intraday_price']).mean()

            # Skip trading for the first row of each day (open period)
            if i == 0:
                trades.append({
                    'date': row['tradeExecTime'],
                    'action': 'day start',
                    'price': row['intraday_price'],
                    'position': position,
                    'capital': current_capital,
                    'acquisition_value': acquisition_value,
                    'avg_acquisition_value': avg_acquisition_value,
                    'trade_pnl': 0,
                    'aggregate_trade_pnl': aggregate_trade_pnl,
                    'predicted_price_difference': price_difference
                })
                previous_price = row['intraday_price']
                continue

            # Check if current capital is greater than zero
            if current_capital <= 0:
                # Close all positions if current capital is not greater than zero
                if position != 0:
                    trade_pnl = position * row['intraday_price'] - acquisition_value - abs(position) * transaction_cost
                    current_capital += position * row['intraday_price'] - abs(position) * transaction_cost
                    aggregate_trade_pnl += trade_pnl
                    trades.append({
                        'date': row['tradeExecTime'],
                        'action': 'close_all',
                        'price': row['intraday_price'],
                        'position': 0,
                        'capital': current_capital,
                        'acquisition_value': 0,
                        'avg_acquisition_value': 0,
                        'trade_pnl': trade_pnl,
                        'aggregate_trade_pnl': aggregate_trade_pnl,
                        'predicted_price_difference': price_difference
                    })
                    position = 0
                    acquisition_value = 0
                    avg_acquisition_value = 0
                    num_trades += 1
                    just_closed = True
                continue

            # Check if previous price was greater than zero
            if previous_price > 0:
                # Stop loss function
                stop_loss_switch = previous_price != 0 and abs(row['intraday_price'] - previous_price) / previous_price >= stop_loss_percentage
                if stop_loss_switch:
                    trade_pnl = position * row['intraday_price'] - acquisition_value - abs(position) * transaction_cost
                    current_capital += position * row['intraday_price'] - abs(position) * transaction_cost
                    aggregate_trade_pnl += trade_pnl
                    trades.append({
                        'date': row['tradeExecTime'],
                        'action': 'stop loss',
                        'price': row['intraday_price'],
                        'position': 0,
                        'capital': current_capital,
                        'acquisition_value': 0,
                        'avg_acquisition_value': 0,
                        'trade_pnl': trade_pnl,
                        'aggregate_trade_pnl': aggregate_trade_pnl,
                        'predicted_price_difference': price_difference
                    })
                    position = 0
                    acquisition_value = 0
                    avg_acquisition_value = 0
                    num_trades += 1
                    just_closed = True

                # Determine the signal
                signal = 1 if price_difference > transaction_cost else (
                    -1 if price_difference < -transaction_cost else 0)

                trade_pnl = 0
                # Adjust position based on signal
                if signal != 0 and not just_closed:
                    # Close existing position if it's opposite to the signal
                    if position * signal < 0:
                        trade_pnl = position * row['intraday_price'] - acquisition_value - abs(
                            position) * transaction_cost
                        current_capital += position * row['intraday_price'] - abs(position) * transaction_cost
                        aggregate_trade_pnl += trade_pnl
                        trades.append({
                            'date': row['tradeExecTime'],
                            'action': 'close',
                            'price': row['intraday_price'],
                            'position': 0,
                            'capital': current_capital,
                            'acquisition_value': 0,
                            'avg_acquisition_value': 0,
                            'trade_pnl': trade_pnl,
                            'aggregate_trade_pnl': aggregate_trade_pnl,
                            'predicted_price_difference': price_difference
                        })
                        position = 0
                        acquisition_value = 0
                        avg_acquisition_value = 0
                        num_trades += 1
                        just_closed = True

                    # Open or add to position only if we didn't just close a position
                    elif not just_closed:
                        new_position = position + signal
                        new_capital = current_capital - transaction_cost - signal * row['intraday_price']
                        if new_capital > 0:
                            current_capital = new_capital
                            acquisition_value += signal * row['intraday_price']
                            avg_acquisition_value = acquisition_value / new_position if new_position != 0 else 0
                            aggregate_trade_pnl -= transaction_cost  # Include transaction cost in aggregate_trade_pnl
                            trades.append({
                                'date': row['tradeExecTime'],
                                'action': 'long' if signal > 0 else 'short',
                                'price': row['intraday_price'],
                                'position': new_position,
                                'capital': current_capital,
                                'acquisition_value': acquisition_value,
                                'avg_acquisition_value': avg_acquisition_value,
                                'trade_pnl': -transaction_cost,
                                'aggregate_trade_pnl': aggregate_trade_pnl,
                                'predicted_price_difference': price_difference
                            })
                            position = new_position
                            num_trades += 1
                else:
                    # Add a record for each row, even if no trade occurred
                    trades.append({
                        'date': row['tradeExecTime'],
                        'action': 'hold',
                        'price': row['intraday_price'],
                        'position': position,
                        'capital': current_capital,
                        'acquisition_value': acquisition_value,
                        'avg_acquisition_value': avg_acquisition_value,
                        'trade_pnl': 0,
                        'aggregate_trade_pnl': aggregate_trade_pnl,
                        'predicted_price_difference': price_difference
                    })
            else:
                # Close any open positions if previous price was not greater than zero
                if position != 0:
                    trade_pnl = position * row['intraday_price'] - acquisition_value - abs(position) * transaction_cost
                    current_capital += position * row['intraday_price'] - abs(position) * transaction_cost
                    aggregate_trade_pnl += trade_pnl
                    trades.append({
                        'date': row['tradeExecTime'],
                        'action': 'close',
                        'price': row['intraday_price'],
                        'position': 0,
                        'capital': current_capital,
                        'acquisition_value': 0,
                        'avg_acquisition_value': 0,
                        'trade_pnl': trade_pnl,
                        'aggregate_trade_pnl': aggregate_trade_pnl,
                        'predicted_price_difference': price_difference
                    })
                    position = 0
                    acquisition_value = 0
                    avg_acquisition_value = 0
                    num_trades += 1
                    just_closed = True
                else:
                    trades.append({
                        'date': row['tradeExecTime'],
                        'action': 'hold',
                        'price': row['intraday_price'],
                        'position': position,
                        'capital': current_capital,
                        'acquisition_value': acquisition_value,
                        'avg_acquisition_value': avg_acquisition_value,
                        'trade_pnl': 0,
                        'aggregate_trade_pnl': aggregate_trade_pnl,
                        'predicted_price_difference': price_difference
                    })

            previous_price = row['intraday_price']
            just_closed = False
            stop_loss_switch = False

        # Close position at the end of the day
        if position != 0:
            last_price = group['intraday_price'].iloc[-1]
            trade_pnl = position * last_price - acquisition_value - abs(position) * transaction_cost
            current_capital += position * last_price - abs(position) * transaction_cost
            aggregate_trade_pnl += trade_pnl
            trades.append({
                'date': group['tradeExecTime'].iloc[-1],
                'action': 'close',
                'price': last_price,
                'position': 0,
                'capital': current_capital,
                'acquisition_value': 0,
                'avg_acquisition_value': 0,
                'trade_pnl': trade_pnl,
                'aggregate_trade_pnl': aggregate_trade_pnl,
                'predicted_price_difference': group['predicted_last_intra_price'].iloc[-1] -
                                              group['predicted_intraday_price'].iloc[-1]
            })
            position = 0
            acquisition_value = 0
            avg_acquisition_value = 0
            num_trades += 1

        # Record daily aggregate
        daily_aggregates.append({
            'date': date,
            'start_capital': start_capital,
            'end_capital': current_capital,
            'change': current_capital - start_capital,
            'num_trades': num_trades
        })

    trades_df = pd.DataFrame(trades)
    daily_aggregates_df = pd.DataFrame(daily_aggregates)

    return trades_df, daily_aggregates_df


# Import the CSV file
base_path = r"C:\Users\chrsr\PycharmProjects\pythonProject\Models and outputs"
save_path = r"C:\Users\chrsr\PycharmProjects\pythonProject\Trading"

trading_run = input("Do you want to run trades? (yes/no): ").lower().strip() == 'yes'

if trading_run:
    transaction_costs = float(input("Transaction cost level? (enter a number): "))

    # Create a time range from 00:00 to 23:30 with 30-minute intervals
    time_range = pd.date_range(start='00:00', end='23:30', freq='30min').strftime('%H:%M')

    # Create the DataFrame <with times and a running integer index starting from 1
    SP_overview_DF = pd.DataFrame({
        'SP': range(1, len(time_range) + 1),
        'Time': time_range
    })

    # Set 'Index' as a column instead of the DataFrame index
    SP_overview_DF.reset_index(drop=True, inplace=True)


    all_cumulative_profit_loss = {}
    all_trades = {}
    use_best_performers = input("Use best performers? (yes/no): ").lower().strip() == 'yes'


    if use_best_performers:
        with open(r"C:\Users\chrsr\PycharmProjects\pythonProject\Trading\best_performers_cost_0.0.txt", 'r') as f:
            best_performers = [int(sp) for sp in f.read().strip().split(',')]
    else:
        print(SP_overview_DF.to_string())
        first = int(input("First SP to run? (enter a number): ")) - 1
        last = int(input("Last SP to run? (enter a number): "))

    trans_costs = transaction_costs

    # Initialize cumulative profit loss DataFrame and all trades DataFrame
    cumulative_profit_loss = pd.DataFrame()
    trades = pd.DataFrame()

    # Round transaction costs to the nearest 0.1
    trans_costs = round(trans_costs * 1000) / 1000


    avg_periods = 10
    cumulative_profit_loss = pd.DataFrame()
    trades = pd.DataFrame()
    trade_pnl = pd.DataFrame()

    portfolio_weights = pd.read_csv(r"C:\Users\chrsr\PycharmProjects\pythonProject\Trading\Portfolio weights v1.csv")

    if use_best_performers:
        sp_values = best_performers
    else:
        sp_values = SP_overview_DF['SP'][first:last]
    for SP_value in sp_values:
        # Get the weight for the current SP from portfolio_weights
        weight_value = portfolio_weights.loc[portfolio_weights['SP'] == SP_value, 'weight'].values[0]
        try:
            SP_weight = float(weight_value)
        except ValueError:
            SP_weight = 0
        # Get the specific start time for this SP
        specific_start = SP_overview_DF[SP_overview_DF['SP'] == SP_value]['Time'].values[0]
        specific_end = (pd.to_datetime(specific_start) + pd.Timedelta(minutes=30)).strftime('%H:%M')
        print(f'running model for SP {SP_value}, transaction cost {trans_costs}, average periods {avg_periods}')
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
        df = df[df['dataset'].isin(['trading'])]

        trades_df, profit_loss_df = long_short_trading_strategy(df, transaction_cost=trans_costs, avg_periods = avg_periods)

        # Add SP column to trades_df
        trades_df['SP'] = SP_value

        # Update cumulative profit loss
        if cumulative_profit_loss.empty:
            cumulative_profit_loss = profit_loss_df[['date', 'change', 'num_trades']].copy()
            cumulative_profit_loss['change'] *= SP_weight
        else:
            temp_df = pd.merge(cumulative_profit_loss, profit_loss_df[['date', 'change', 'num_trades']], on='date',
                               how='outer', suffixes=('', '_new'))
            cumulative_profit_loss = pd.DataFrame({
                'date': temp_df['date'],
                'change': temp_df['change'].ffill().add(temp_df['change_new'].ffill().multiply(SP_weight), fill_value=0),
                'num_trades': temp_df['num_trades'].ffill().add(temp_df['num_trades_new'].ffill(), fill_value=0)
            })

        # Append to all trades DataFrame
        # Create a DataFrame with date, SP, and trade_pnl for the current trades_df
        current_sp_df = pd.DataFrame({
            'date': trades_df['date'],
            f'SP_{SP_value}': trades_df['aggregate_trade_pnl']
        })

        # If trade_pnl DataFrame doesn't exist, create it
        if trade_pnl.empty:
            trade_pnl = current_sp_df
        else:
            # Merge with existing trade_pnl DataFrame
            trade_pnl = pd.merge(trade_pnl, current_sp_df, on='date', how='outer')
            # Fill blank values with the last non-null value for each column
            trade_pnl = trade_pnl.ffill()

        trades = pd.concat([trades, trades_df], ignore_index=True)

    # Sort cumulative profit loss by date
    cumulative_profit_loss = cumulative_profit_loss.sort_values('date')
    # Add a new column for aggregated change
    cumulative_profit_loss['aggregated_change'] = cumulative_profit_loss['change'].cumsum()

    save_suffix = str(input("Identifier for saved file? (e.g. cumulative_profit_loss_cost_xxx): "))

    # Save cumulative profit loss and all trades to CSV files
    cumulative_profit_loss.to_csv(os.path.join(save_path, f'cumulative_profit_loss_cost_{save_suffix}.csv'),
                                  index=False)
    trades.to_csv(os.path.join(save_path, f'all_trades_cost_{save_suffix}.csv'), index=False)
    trade_pnl.to_csv(os.path.join(save_path, f'trade_pnl_cost_{save_suffix}.csv'), index=False)
    print(f'saved profit and loss as cumulative_profit_loss_cost_{save_suffix}.csv')
    print(f'saved trades as all_trades_cost_{save_suffix}.csv')
