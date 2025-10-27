import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ardl import ARDL
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import itertools

# 1. Kiểm tra dữ liệu đầu vào
# Đọc dữ liệu
brent_df = pd.read_excel("FRED.xlsx")
ron95_df = pd.read_excel("ron95_price_updated_ver3.xlsx")
cpi_df = pd.read_csv("vietnam_cpi_2019_2025.csv")

# In một vài dòng để kiểm tra
print("Brent Data (FRED.xlsx):")
print(brent_df.head())
print("\nRON95 Data (ron95_price_updated_ver3.xlsx):")
print(ron95_df.head())
print("\nCPI Data (vietnam_cpi_2019_2025.csv):")
print(cpi_df.head())

# Kiểm tra giá trị thiếu
print("\nMissing values in brent_df:", brent_df.isnull().sum())
print("Missing values in ron95_df:", ron95_df.isnull().sum())
print("Missing values in cpi_df:", cpi_df.isnull().sum())

# Kiểm tra kiểu dữ liệu
print("\nData types in brent_df:", brent_df.dtypes)
print("Data types in ron95_df:", ron95_df.dtypes)
print("Data types in cpi_df:", cpi_df.dtypes)

# Convert dates to datetime
if brent_df['observation_date'].dtype != 'datetime64[ns]':
    brent_df['observation_date'] = pd.to_timedelta(brent_df['observation_date'], unit='d') + datetime(1899, 12, 30)
if ron95_df['Date'].dtype != 'datetime64[ns]':
    ron95_df['Date'] = pd.to_datetime(ron95_df['Date'], origin='1899-12-30', unit='D')
cpi_df['Date'] = pd.to_datetime(cpi_df['Date'])

# Kiểm tra sự khớp nối thời gian
print("\nDate range in brent_df:", brent_df['observation_date'].min(), "to", brent_df['observation_date'].max())
print("Date range in ron95_df:", ron95_df['Date'].min(), "to", ron95_df['Date'].max())
print("Date range in cpi_df:", cpi_df['Date'].min(), "to", cpi_df['Date'].max())

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
missing_cols = [col for col in required_columns if col not in merged_df.columns]
if missing_cols:
    raise KeyError(f"Missing columns in merged_df: {missing_cols}")

data = merged_df[required_columns].copy()

# Tính Price_volatility
data['Price_volatility'] = data['Price'].rolling(window=5).std()

# 2. Xử lý giá trị thiếu và căn chỉnh dữ liệu
# Xóa các hàng có giá trị thiếu trong các cột cần thiết
data = data.dropna()
print("\nData after dropping NaN:")
print(data.head())
print("Data index:", data.index)

# 3. Kiểm tra tính dừng (ADF Test)
def adf_test(series, name):
    result = adfuller(series.dropna())
    print(f'\nADF Test for {name}:')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Stationary' if result[1] < 0.05 else 'Non-Stationary')
    return result[1] < 0.05

# Danh sách các biến cần kiểm tra
variables = ['Price', 'Price_volatility', 'Quantity', 'QBOG_balance', 'POILBREUSDM', 'CPI_chung', 'CPI_giao_thong', 'Income']
stationarity_results = {}
for var in variables:
    stationarity_results[var] = adf_test(data[var], var)

# Xử lý các biến không dừng (nếu cần)
for var, is_stationary in stationarity_results.items():
    if not is_stationary:
        print(f"{var} is non-stationary, applying first differencing...")
        data[f'{var}_diff'] = data[var].diff()

# Cập nhật data với các biến sai phân nếu cần
use_diff = False
if not all(stationarity_results.values()):
    use_diff = True
    exog_columns = [col + '_diff' if col in stationarity_results and not stationarity_results[col] else col 
                    for col in ['QBOG_balance', 'POILBREUSDM', 'Inflation_rate', 'CPI_chung', 'CPI_giao_thong', 'Income']]
    endog_pv = data['Price_volatility_diff'] if not stationarity_results['Price_volatility'] else data['Price_volatility']
    endog_qty = data['Quantity_diff'] if not stationarity_results['Quantity'] else data['Quantity']
else:
    exog_columns = ['QBOG_balance', 'POILBREUSDM', 'Inflation_rate', 'CPI_chung', 'CPI_giao_thong', 'Income']
    endog_pv = data['Price_volatility']
    endog_qty = data['Quantity']

# 4. Kiểm tra đa cộng tuyến
exog = data[exog_columns].dropna()
print("\nExog after dropping NaN:")
print(exog.head())
print("Exog index:", exog.index)

# Ma trận tương quan
correlation_matrix = exog.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Tính VIF
vif_data = pd.DataFrame()
vif_data['Variable'] = exog.columns
vif_data['VIF'] = [variance_inflation_factor(exog.values, i) for i in range(exog.shape[1])]
print("\nVIF Values:")
print(vif_data)

# Loại bỏ các biến có VIF > 10
high_vif_vars = vif_data[vif_data['VIF'] > 10]['Variable'].tolist()
if high_vif_vars:
    print(f"Removing high VIF variables: {high_vif_vars}")
    exog = exog.drop(columns=high_vif_vars)
    exog_columns = [col for col in exog_columns if col not in high_vif_vars]
else:
    print("No variables with high VIF (>10).")

# Chuẩn hóa dữ liệu, giữ nguyên chỉ số
scaler = StandardScaler()
exog_scaled = pd.DataFrame(scaler.fit_transform(exog), columns=exog.columns, index=exog.index)
print("\nExog_scaled after standardization:")
print(exog_scaled.head())
print("Exog_scaled index:", exog_scaled.index)

# 5. Tối ưu hóa số độ trễ
def select_optimal_lags(endog, exog, maxlag, maxorder, ic='aic'):
    best_ic = np.inf
    best_lags = None
    best_order = None
    for lags in range(1, maxlag + 1):
        for order in itertools.product(range(maxorder + 1), repeat=len(exog.columns)):
            try:
                model = ARDL(endog, lags=lags, exog=exog, order=order, trend='c')
                results = model.fit()
                ic_value = results.aic if ic == 'aic' else results.bic
                if ic_value < best_ic:
                    best_ic = ic_value
                    best_lags = lags
                    best_order = order
            except:
                continue
    return best_lags, best_order

# Set Date as index if not already set
if not isinstance(data.index, pd.DatetimeIndex):
    data.set_index('Date', inplace=True)
    data.index = pd.DatetimeIndex(data.index, freq=None)

# ARDL model for Price_volatility
endog_pv = endog_pv.loc[exog_scaled.index]  # Align endog with exog_scaled
print("\nEndog_pv index:", endog_pv.index)

# Chọn số độ trễ tối ưu
print("\nSelecting optimal lags for Price_volatility model...")
optimal_lags_pv, optimal_order_pv = select_optimal_lags(endog_pv, exog_scaled, maxlag=4, maxorder=4, ic='aic')
print(f"Optimal lags for Price_volatility: {optimal_lags_pv}")
print(f"Optimal order for exogenous variables: {optimal_order_pv}")

# Chạy lại mô hình ARDL với số độ trễ tối ưu
if optimal_lags_pv is not None and optimal_order_pv is not None:
    model_pv = ARDL(endog_pv, lags=optimal_lags_pv, exog=exog_scaled, order=optimal_order_pv, trend='c')
    results_pv = model_pv.fit()
    results_pv_summary = results_pv.summary().tables[1].as_csv()
    with open('ardl_price_volatility_results_updated.csv', 'w') as f:
        f.write(results_pv_summary)
    print("\nParameters in updated Price_volatility model:", results_pv.params.index.tolist())

    # Tính long-run coefficients
    lag_coefs_pv = sum([results_pv.params.get(f'Price_volatility.L{i}', 0) for i in range(1, optimal_lags_pv + 1)])
    long_run_pv = results_pv.params[exog_scaled.columns] / (1 - lag_coefs_pv)
    print("\nLong-run coefficients for Price_volatility:")
    print(long_run_pv)
else:
    print("No optimal lags found for Price_volatility model.")

# ARDL model for Quantity
endog_qty = endog_qty.loc[exog_scaled.index]  # Align endog with exog_scaled
print("\nEndog_qty index:", endog_qty.index)

# Chọn số độ trễ tối ưu
print("\nSelecting optimal lags for Quantity model...")
optimal_lags_qty, optimal_order_qty = select_optimal_lags(endog_qty, exog_scaled, maxlag=4, maxorder=4, ic='aic')
print(f"Optimal lags for Quantity: {optimal_lags_qty}")
print(f"Optimal order for exogenous variables: {optimal_order_qty}")

# Chạy lại mô hình ARDL với số độ trễ tối ưu
if optimal_lags_qty is not None and optimal_order_qty is not None:
    model_qty = ARDL(endog_qty, lags=optimal_lags_qty, exog=exog_scaled, order=optimal_order_qty, trend='c')
    results_qty = model_qty.fit()
    results_qty_summary = results_qty.summary().tables[1].as_csv()
    with open('ardl_quantity_results_updated.csv', 'w') as f:
        f.write(results_qty_summary)
    print("\nParameters in updated Quantity model:", results_qty.params.index.tolist())

    # Tính long-run coefficients
    lag_coefs_qty = sum([results_qty.params.get(f'Quantity.L{i}', 0) for i in range(1, optimal_lags_qty + 1)])
    long_run_qty = results_qty.params[exog_scaled.columns] / (1 - lag_coefs_qty)
    print("\nLong-run coefficients for Quantity:")
    print(long_run_qty)
else:
    print("No optimal lags found for Quantity model.")

# 6. Phân tích kết quả QBOG_balance
analysis_results = []
if 'QBOG_balance.L0' in results_pv.params.index:
    qbog_coeff_pv = results_pv.params['QBOG_balance.L0']
    qbog_pvalue_pv = results_pv.pvalues['QBOG_balance.L0']
    print(f"\nCoefficient of QBOG_balance.L0 (Price_volatility): {qbog_coeff_pv}, p-value: {qbog_pvalue_pv}")
    analysis_results.append(f"Coefficient of QBOG_balance.L0 (Price_volatility): {qbog_coeff_pv}, p-value: {qbog_pvalue_pv}")
    if qbog_coeff_pv < 0 and qbog_pvalue_pv < 0.05:
        analysis_results.append("QBOG_balance significantly reduces price volatility.")
    else:
        analysis_results.append("QBOG_balance does not significantly reduce price volatility.")

if 'QBOG_balance.L0' in results_qty.params.index:
    qbog_coeff_qty = results_qty.params['QBOG_balance.L0']
    qbog_pvalue_qty = results_qty.pvalues['QBOG_balance.L0']
    print(f"Coefficient of QBOG_balance.L0 (Quantity): {qbog_coeff_qty}, p-value: {qbog_pvalue_qty}")
    analysis_results.append(f"Coefficient of QBOG_balance.L0 (Quantity): {qbog_coeff_qty}, p-value: {qbog_pvalue_qty}")
    if qbog_coeff_qty > 0 and qbog_pvalue_qty < 0.05:
        analysis_results.append("QBOG_balance significantly stabilizes quantity (increases consumption).")
    else:
        analysis_results.append("QBOG_balance does not significantly stabilize quantity.")

with open('analysis_results_updated.txt', 'w') as f:
    f.write("\n".join(analysis_results))

# 7. Vẽ biểu đồ để trực quan hóa
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['QBOG_balance'], label='QBOG_balance')
plt.plot(data.index, data['Price_volatility'], label='Price Volatility')
plt.title('QBOG_balance vs Price Volatility')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.savefig('qbog_vs_volatility.png')
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['QBOG_balance'], label='QBOG_balance')
plt.plot(data.index, data['Quantity'], label='Quantity')
plt.title('QBOG_balance vs Quantity')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.savefig('qbog_vs_quantity.png')
plt.close()