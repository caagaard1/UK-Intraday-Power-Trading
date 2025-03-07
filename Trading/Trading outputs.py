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

save_path = r"C:\Users\chrsr\PycharmProjects\pythonProject\Trading"

if input("Do you want to showcase info for a specific file (yes/no): ").lower().strip() == 'yes':
    Specific_file = str(input("Identifier for file? (e.g. cumulative_profit_loss_cost_xxx): "))
    cumulative_profit_loss = pd.read_csv(os.path.join(save_path, f'cumulative_profit_loss_cost_{Specific_file}.csv'))
    trades = pd.read_csv(os.path.join(save_path, f'all_trades_cost_{Specific_file}.csv'))
    trade_pnl = pd.read_csv(os.path.join(save_path, f'trade_pnl_cost_{Specific_file}.csv'))

    # Determine the number of unique values in the 'SP' column of trades DataFrame
    num_unique_sp = trades['SP'].nunique()

    # Set total portfolio value
    portfolio_value_beg = num_unique_sp * 1000

    # Create new 'capital' column as sum of initial portfolio value and aggregated change
    cumulative_profit_loss['capital'] = portfolio_value_beg + cumulative_profit_loss['aggregated_change']
    cumulative_profit_loss['capital_rebased'] = cumulative_profit_loss['capital'] / portfolio_value_beg * 100
    cumulative_profit_loss['percentage_change'] = np.where(
        cumulative_profit_loss['capital'].shift(1) != 0,
        (cumulative_profit_loss['capital'] - cumulative_profit_loss['capital'].shift(1)) / cumulative_profit_loss[
            'capital'].shift(1) * 100,
        0
    )

    if input("Do you want to plot cumulative performance of all models (yes/no): ").lower().strip() == 'yes':
        print(cumulative_profit_loss.head(30).to_string())
        print(trades.head(30).to_string())
        print(trades.shape)
        print(cumulative_profit_loss.shape)

        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # Create a figure and axis objects
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot the aggregated_change as a line
        ax1.plot(cumulative_profit_loss['date'], cumulative_profit_loss['capital_rebased'], color='blue',
                 label='Capital development')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Capital index = 100', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(0, 200)  # Set y-axis range from 0 to 150

        # Create a secondary y-axis
        ax2 = ax1.twinx()

        # Plot the change as bars with green for positive and red for negative
        colors = ['green' if x >= 0 else 'red' for x in cumulative_profit_loss['change']]
        ax2.bar(cumulative_profit_loss['date'], cumulative_profit_loss['percentage_change'], alpha=0.3, color=colors,
                label='Change')
        ax2.set_ylabel('Change per day (%)', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: '{:.1f}%'.format(y)))  # Format y-axis as percentages

        # Format x-axis to show dates nicely
        # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()  # Rotate and align the tick labels

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title('Cumulative Profit Loss: Capital Development and Daily Change')
        plt.tight_layout()
        plt.show()

    if input("Do you want to plot individual performance of all models and save positive performers (yes/no): ").lower().strip() == 'yes':
        # Ensure the date column is in datetime format
        trade_pnl['date'] = pd.to_datetime(trade_pnl['date'])

        # Get columns with the 20 highest last values (excluding 'date')
        numeric_columns = trade_pnl.select_dtypes(include=[np.number]).columns
        last_values = trade_pnl[numeric_columns].iloc[-1]
        top_20_columns = last_values.nlargest(20).index.tolist()

        # Plot individual performance for top 20 columns
        plt.figure(figsize=(12, 6))

        for column in top_20_columns:
            plt.plot(trade_pnl['date'], trade_pnl[column], label=column)

        plt.xlabel('Date')
        plt.ylabel('Cumulative P&L')
        plt.title('Individual Performance of Top 20 Models')
        plt.legend(title='SP', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Format x-axis to show only dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()  # Rotate and align the tick labels

        plt.tight_layout()
        plt.show()

        # Get columns with positive last values (excluding 'date')
        positive_columns = last_values[last_values > 0].index.tolist()

        # Save best performers as a list of SP values
        if input("Save best performers (yes/no): ").lower().strip() == "yes":
            best_performers = [int(col.split('_')[-1]) for col in positive_columns]
            with open(os.path.join(save_path, f'best_performers_cost_{Specific_file}.txt'), 'w') as f:
                f.write(','.join(map(str, best_performers)))
            print(f"Best performers saved to 'best_performers_cost_{Specific_file}.txt'")


    if input("Do you want to plot cumulative performance of all models and variations (yes/no): ").lower().strip() == 'yes':
        # Create a figure and axis objects
        fig, ax1 = plt.subplots(figsize=(12, 6))
        cost = "0.07"

        for i in np.arange(1,11,1):
            cumulative_profit_loss = pd.read_csv(
                os.path.join(save_path, fr'C:\Users\chrsr\PycharmProjects\pythonProject\Trading\cumulative_profit_loss_cost_{cost}_{i}.csv'))

            # Create new 'capital' column as sum of initial portfolio value and aggregated change
            cumulative_profit_loss['capital'] = portfolio_value_beg + cumulative_profit_loss['aggregated_change']
            cumulative_profit_loss['capital_rebased'] = cumulative_profit_loss['capital'] / portfolio_value_beg * 100
            cumulative_profit_loss['percentage_change'] = np.where(
                cumulative_profit_loss['capital'].shift(1) != 0,
                (cumulative_profit_loss['capital'] - cumulative_profit_loss['capital'].shift(1)) / cumulative_profit_loss[
                    'capital'].shift(1) * 100,0)

            # Plot the aggregated_change as a line
            ax1.plot(cumulative_profit_loss['date'], cumulative_profit_loss['capital_rebased'], label=f"{i}")


        ax1.set_xlabel('Date')
        ax1.set_ylabel('Capital index = 100', color='blue')
        ax1.set_ylim(50, 200)  # Set y-axis range from 0 to 150
        ax1.axhline(y=100, color='black', linestyle=':', label='Baseline')

        # Format x-axis to show dates nicely
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()  # Rotate and align the tick labels

        # Add legend
        plt.legend(title='avg_periods', bbox_to_anchor = (1.05, 1),  loc='upper left')

        plt.title('Cumulative Profit Loss: Capital Development and Daily Change')
        plt.tight_layout()
        plt.show()