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


save_suffix = str(input("Identifier for saved file? (e.g. cumulative_profit_loss_cost_xxx): "))

cumulative_profit_loss = pd.read_csv(os.path.join( rf'C:\Users\chrsr\PycharmProjects\pythonProject\Trading\cumulative_profit_loss_cost_{save_suffix}.csv'))
trades = pd.read_csv(os.path.join(rf'C:\Users\chrsr\PycharmProjects\pythonProject\Trading\all_trades_cost_{save_suffix}.csv'))
trade_pnl = pd.read_csv(os.path.join(rf'C:\Users\chrsr\PycharmProjects\pythonProject\Trading\trade_pnl_cost_{save_suffix}.csv'))

# Create a figure and axis objects
fig, ax1 = plt.subplots(figsize=(12, 6))

# Convert 'date' column to datetime
cumulative_profit_loss['date'] = pd.to_datetime(cumulative_profit_loss['date'])

# Plot the aggregated_change as a line
ax1.plot(cumulative_profit_loss['date'], cumulative_profit_loss['aggregated_change'] + 1000, color='blue',
         label='Capital development')
ax1.set_xlabel('Date')
ax1.set_ylabel('Start capital = 1000', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Calculate the percentage change
last_value = cumulative_profit_loss['aggregated_change'].iloc[-1] + 1000
percentage_change = ((last_value - 1000) / 1000) * 100

# Add a call out box for the last value
last_date = cumulative_profit_loss['date'].iloc[-1]
sign = '+' if percentage_change >= 0 else '-'

ax1.annotate(f'{sign}{abs(percentage_change):.2f}%',
             xy=(last_date, last_value),
             xytext=(10, 10), textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Format x-axis to show dates nicely
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()  # Rotate and align the tick labels

plt.title(f'Cumulative Profit Loss: Capital Development - {save_suffix}')
plt.legend()
plt.tight_layout()
plt.show()
