import pandas as pd
import numpy as np
from statsmodels.tsa.api import ARDL
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta

# Function to convert Excel serial date to datetime
def excel_date_to_datetime(excel_date):
    if isinstance(excel_date, (int, float)):
        return datetime(1899, 12, 30) + timedelta(days=excel_date)
    return excel_date  # Return as-is if already a datetime/Timestamp

# Load the datasets
ron95_data = pd.read_excel('ron95_price_updated_ver2.xlsx')
brent_data = pd.read_excel('FRED.xlsx')

# Check and convert date columns
if ron95_data['Date'].dtype in ['int64', 'float64']:
    ron95_data['Date'] = ron95_data['Date'].apply(excel_date_to_datetime)
else:
    ron95_data['Date'] = pd.to_datetime(ron95_data['Date'])

if brent_data['observation_date'].dtype in ['int64', 'float64']:
    brent_data['observation_date'] = brent_data['observation_date'].apply(excel_date_to_datetime)
else:
    brent_data['observation_date'] = pd.to_datetime(brent_data['observation_date'])

# Rename columns for consistency
ron95_data = ron95_data.rename(columns={'Date': 'date', 'Price': 'retail_price', 'ER_daily': 'exchange_rate'})
brent_data = brent_data.rename(columns={'observation_date': 'date', 'POILBREUSDM': 'brent_price_usd'})

# Merge datasets on date
data = pd.merge_asof(
    ron95_data[['date', 'retail_price', 'exchange_rate', 'Inflation_rate', 'Quantity']],
    brent_data[['date', 'brent_price_usd']],
    on='date',
    direction='nearest',
    tolerance=pd.Timedelta('7 days')
)

# Add tax_dummy column based on date (1 for before 01/07/2025, 0 for after)
data['tax_dummy'] = data['date'].apply(lambda x: 1 if x < pd.Timestamp('2025-07-01') else 0)

# Select relevant columns for analysis
data = data[['date', 'retail_price', 'brent_price_usd', 'exchange_rate', 'Inflation_rate',]]

# Print descriptive statistics for exchange_rate to check variability
print("Descriptive Statistics for exchange_rate:")
print(data['exchange_rate'].describe())
print("\nUnique values in exchange_rate:")
print(data['exchange_rate'].value_counts())

# Check stationarity with ADF test
def check_stationarity(series, name):
    result = adfuller(series.dropna(), autolag='AIC')
    print(f'ADF Test for {name}:')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values: {result[4]}\n')

for column in ['retail_price', 'brent_price_usd', 'exchange_rate', 'Inflation_rate']:
    check_stationarity(data[column], column)

# Log-transform variables where appropriate
data['log_retail_price'] = np.log(data['retail_price'])
data['log_brent_price_usd'] = np.log(data['brent_price_usd'])
data['log_exchange_rate'] = np.log(data['exchange_rate'])
# data['log_quantity'] = np.log(data['Quantity'])  # Log for Quantity as it's positive
# Keep QBOG_balance (can be negative), Inflation_rate (rate), Holiday_dummy (0/1), tax_dummy (0/1) as is

# Set date as index
data.set_index('date', inplace=True)

# Define ARDL model with brent_price_usd and tax_dummy
ardl_model = ARDL(
    endog=data['log_retail_price'],
    exog=data[['log_brent_price_usd', 'log_exchange_rate','Inflation_rate']],
    lags=4,  # Based on previous model and weekly adjustment
    order=4,  # Distributed lags for exogenous variables
    trend='c'  # Include constant
)

# Fit the model
ardl_results = ardl_model.fit()

# Print model summary
print(ardl_results.summary())

# Compute long-run coefficients (pass-through elasticities)
ar_coeff_sum = sum(ardl_results.params.get(f'log_retail_price.L{j}', 0) for j in range(1, 5))
denominator = 1 - ar_coeff_sum

# For each exogenous variable, sum the coefficients (including L0)
exog_vars = ['log_brent_price_usd', 'log_exchange_rate', 'Inflation_rate']
long_run_coeffs = {}
for var in exog_vars:
    coeff_sum = sum(ardl_results.params.get(f'{var}.L{i}', 0) for i in range(5))
    long_run_coeffs[var] = coeff_sum / denominator if denominator != 0 else np.nan

print("\nLong-run Pass-through Coefficients:")
for var, coeff in long_run_coeffs.items():
    print(f"{var}: {coeff:.4f}")

# Determine lag structure (number of significant lags)
significant_lags_keys = []
for var in exog_vars:
    significant_lags_keys += [f'{var}.L{i}' for i in range(1, 5)]  # Exclude L0 as it's contemporaneous

significant_lags = ardl_results.pvalues[significant_lags_keys]
lag_periods = sum(p < 0.05 for p in significant_lags)
print(f"\nNumber of significant lags for price adjustment: {lag_periods} periods (approximately {lag_periods*7} days)")

# Save results to a file
with open('RQ2/ardl_results_with_tax_dummy.txt', 'w') as f:
    f.write(str(ardl_results.summary()))
    f.write("\n\nLong-run Pass-through Coefficients:\n")
    for var, coeff in long_run_coeffs.items():
        f.write(f"{var}: {coeff:.4f}\n")
    f.write(f"Number of significant lags: {lag_periods} periods (approximately {lag_periods*7} days)\n")