import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ardl import ARDL
import matplotlib.pyplot as plt

# Read data
brent_df = pd.read_excel("FRED.xlsx")
ron95_df = pd.read_excel("ron95_price_updated_ver3.xlsx")  # Sử dụng file mới
cpi_df = pd.read_csv("vietnam_cpi_2019_2025.csv")  # Read CPI data

# Convert dates to datetime if needed
if brent_df['observation_date'].dtype != 'datetime64[ns]':
    brent_df['observation_date'] = pd.to_timedelta(brent_df['observation_date'], unit='d') + datetime(1899, 12, 30)
if ron95_df['Date'].dtype != 'datetime64[ns]':
    ron95_df['Date'] = pd.to_datetime(ron95_df['Date'], origin='1899-12-30', unit='D')
cpi_df['Date'] = pd.to_datetime(cpi_df['Date'])  # Convert CPI Date to datetime

# Merge data
merged_df = pd.merge_asof(
    ron95_df.sort_values('Date'),
    brent_df.sort_values('observation_date'),
    left_on='Date',
    right_on='observation_date',
    direction='nearest'
)
merged_df = pd.merge_asof(
    merged_df.sort_values('Date'),
    cpi_df.sort_values('Date'),
    on='Date',
    direction='nearest'
)

# Select required columns
required_columns = ['Date', 'Price', 'Quantity', 'QBOG_balance', 'POILBREUSDM', 'ER_daily', 'Inflation_rate', 'Holiday_dummy', 'CPI_chung', 'CPI_giao_thong', 'Income']
if not all(col in merged_df.columns for col in required_columns):
    missing_cols = [col for col in required_columns if col not in merged_df.columns]
    raise KeyError(f"Missing columns in merged_df: {missing_cols}")

data = merged_df[required_columns].copy()

# Calculate price volatility
data['Price_volatility'] = data['Price'].rolling(window=5).std()

# ADF test (optional, for reference)
def adf_test(series, name):
    result = adfuller(series.dropna())
    print(f'ADF Test for {name}:')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Stationary' if result[1] < 0.05 else 'Non-Stationary')

adf_test(data['Price'], 'Price')
adf_test(data['Price_volatility'], 'Price Volatility')
adf_test(data['Quantity'], 'Quantity')
adf_test(data['QBOG_balance'], 'QBOG_balance')
adf_test(data['POILBREUSDM'], 'POILBREUSDM')
adf_test(data['CPI_chung'], 'CPI_chung')
adf_test(data['CPI_giao_thong'], 'CPI_giao_thong')
adf_test(data['Income'], 'Income')  # Thêm kiểm tra tính dừng cho Income

# Handle missing values
data = data.dropna()

# Set Date as index
data.set_index('Date', inplace=True)
data.index = pd.DatetimeIndex(data.index, freq=None)

# ARDL model for Price_volatility
endog = data['Price_volatility']
exog = data[['QBOG_balance', 'POILBREUSDM', 'Inflation_rate', 'CPI_chung', 'CPI_giao_thong', 'Income', 'Holiday_dummy']]  # Thêm Income
model = ARDL(endog, lags=4, exog=exog, order=4, trend='c')
results = model.fit()
results_summary = results.summary().tables[1].as_csv()
with open('ardl_price_volatility_results.csv', 'w') as f:
    f.write(results_summary)
print("Parameters in Price_volatility model:", results.params.index.tolist())

analysis_results = []
if 'QBOG_balance.L0' in results.params.index:
    qbog_coeff = results.params['QBOG_balance.L0']
    print(f"Coefficient of QBOG_balance.L0: {qbog_coeff}")
    analysis_results.append(f"Coefficient of QBOG_balance.L0 (Price_volatility): {qbog_coeff}")
    if qbog_coeff < 0:
        analysis_results.append("QBOG_balance reduces price volatility.")
    else:
        analysis_results.append("QBOG_balance does not significantly reduce price volatility or may increase it.")

# ARDL model for Quantity
endog_qty = data['Quantity']
exog_qty = data[['QBOG_balance',  'CPI_chung', 'CPI_giao_thong', 'Income']]  # Thêm Income
model_qty = ARDL(endog_qty, lags=4, exog=exog_qty, order=4, trend='c')
results_qty = model_qty.fit()
results_qty_summary = results_qty.summary().tables[1].as_csv()
with open('ardl_quantity_results.csv', 'w') as f:
    f.write(results_qty_summary)
print("Parameters in Quantity model:", results_qty.params.index.tolist())

if 'QBOG_balance.L0' in results_qty.params.index:
    qbog_coeff_qty = results_qty.params['QBOG_balance.L0']
    print(f"Coefficient of QBOG_balance.L0 for Quantity: {qbog_coeff_qty}")
    analysis_results.append(f"Coefficient of QBOG_balance.L0 (Quantity): {qbog_coeff_qty}")
    if qbog_coeff_qty > 0:
        analysis_results.append("QBOG_balance stabilizes quantity (increases consumption).")
    else:
        analysis_results.append("QBOG_balance does not significantly stabilize quantity or may reduce it.")

with open('analysis_results.txt', 'w') as f:
    f.write("\n".join(analysis_results))

# Plot and save price volatility
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Price_volatility'], label='Price Volatility')
plt.title('RON95 Price Volatility Over Time')
plt.xlabel('Date')
plt.ylabel('Price Volatility')
plt.legend()
plt.savefig('price_volatility_plot.png')
plt.close()

# Plot and save quantity
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Quantity'], label='Quantity', color='orange')
plt.title('RON95 Quantity Over Time')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.legend()
plt.savefig('quantity_plot.png')
plt.close()