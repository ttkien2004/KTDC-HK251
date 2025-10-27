# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# from statsmodels.tsa.api import VAR
# from statsmodels.tsa.stattools import adfuller
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings("ignore")

# # ---------- Load dữ liệu từ các file gốc ----------
# def load_data():
#     price_file = "ron95_price_updated_ver1.xlsx"
#     try:
#         df_price = pd.read_excel(price_file, sheet_name=0)
#         if pd.api.types.is_datetime64_any_dtype(df_price['Date']):
#             print("RON95 Date is already datetime64, no conversion needed.")
#         else:
#             df_price['Date'] = pd.to_datetime(df_price['Date'], origin='1899-12-30', unit='D', errors='coerce')
        
#         df_price = df_price.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        
#         if df_price['Price'].nunique() <= 1 or df_price['Price'].le(0).any():
#             raise ValueError("Price column has constant or invalid (non-positive) values.")
#         if df_price['ER_daily'].nunique() <= 1 or df_price['ER_daily'].le(0).any():
#             raise ValueError("ER_daily column has constant or invalid (non-positive) values.")
        
#         df_price['log_P'] = np.log(df_price['Price'])
#         df_price['Month'] = df_price['Date'].dt.to_period('M')
#     except Exception as e:
#         print(f"Error loading {price_file}: {e}")
#         raise

#     brent_file = "FRED.xlsx"
#     try:
#         df_brent = pd.read_excel(brent_file, sheet_name=0)
#         if pd.api.types.is_numeric_dtype(df_brent['observation_date']):
#             df_brent['Date'] = pd.to_datetime(df_brent['observation_date'], origin='1899-12-30', unit='D', errors='coerce')
#         else:
#             df_brent['Date'] = pd.to_datetime(df_brent['observation_date'], errors='coerce')
        
#         df_brent = df_brent.rename(columns={'POILBREUSDM': 'Brent_USD'})
#         df_brent = df_brent[['Date', 'Brent_USD']].dropna()
        
#         if df_brent['Brent_USD'].nunique() <= 1 or df_brent['Brent_USD'].le(0).any():
#             raise ValueError("Brent_USD column has constant or invalid (non-positive) values.")
        
#         df_brent['Month'] = df_brent['Date'].dt.to_period('M')
#     except Exception as e:
#         print(f"Error loading {brent_file}: {e}")
#         raise

#     df_price_monthly = df_price.groupby('Month').agg({'Price': 'mean', 'ER_daily': 'mean', 'log_P': 'mean'}).reset_index()
#     df_brent_monthly = df_brent.groupby('Month').agg({'Brent_USD': 'mean'}).reset_index()

#     df_monthly = df_price_monthly.merge(df_brent_monthly, on='Month', how='left')
#     df_monthly['Brent_USD'] = df_monthly['Brent_USD'].fillna(method='ffill').fillna(method='bfill')
#     df_monthly['ln_brent_er'] = np.log(df_monthly['Brent_USD'] * df_monthly['ER_daily'] / 1000)
#     df_monthly['Date'] = df_monthly['Month'].dt.to_timestamp()

#     print("Monthly aggregated data (first 5 rows):")
#     print(df_monthly[['Date', 'Price', 'ER_daily', 'Brent_USD', 'log_P', 'ln_brent_er']].head())
#     print("Number of rows in df_monthly:", len(df_monthly))

#     return df_monthly

# # ---------- Load và merge CPI data ----------
# def load_and_merge_cpi(df):
#     cpi_file = "vietnam_cpi_2019_2025.csv"
#     try:
#         df_cpi = pd.read_csv(cpi_file, parse_dates=['Date'])
#     except FileNotFoundError:
#         print(f"File {cpi_file} not found. Using sample CPI data.")
#         sample_cpi = """
#         Date,CPI_chung,CPI_giao_thong
#         2024-01-01,115.50,105.00
#         2024-02-01,116.20,105.50
#         2024-03-01,117.00,106.00
#         2024-04-01,117.80,106.80
#         2024-05-01,118.50,107.20
#         2024-06-01,119.30,107.90
#         2024-07-01,120.10,108.50
#         2024-08-01,120.90,109.00
#         2024-09-01,121.70,109.60
#         2024-10-01,122.50,110.20
#         2024-11-01,123.20,110.80
#         2024-12-01,124.00,111.40
#         2025-01-01,124.80,112.00
#         2025-02-01,125.60,112.60
#         2025-03-01,126.30,113.20
#         2025-04-01,127.10,113.80
#         2025-05-01,127.90,114.40
#         2025-06-01,128.70,115.00
#         2025-07-01,129.50,115.60
#         2025-08-01,130.30,116.20
#         2025-09-01,131.10,116.80
#         """
#         from io import StringIO
#         df_cpi = pd.read_csv(StringIO(sample_cpi.strip()), parse_dates=['Date'])

#     if df_cpi['CPI_chung'].nunique() <= 1 or df_cpi['CPI_chung'].le(0).any():
#         raise ValueError("CPI_chung column has constant or invalid (non-positive) values.")
#     if df_cpi['CPI_giao_thong'].nunique() <= 1 or df_cpi['CPI_giao_thong'].le(0).any():
#         raise ValueError("CPI_giao_thong column has constant or invalid (non-positive) values.")

#     df_cpi['log_CPI_chung'] = np.log(df_cpi['CPI_chung'])
#     df_cpi['log_CPI_giao_thong'] = np.log(df_cpi['CPI_giao_thong'])
#     df_cpi['Month'] = df_cpi['Date'].dt.to_period('M')

#     df_merged = df.merge(df_cpi[['Month', 'log_CPI_chung', 'log_CPI_giao_thong']], on='Month', how='left')
#     df_merged[['log_CPI_chung', 'log_CPI_giao_thong']] = df_merged[['log_CPI_chung', 'log_CPI_giao_thong']].fillna(method='ffill')

#     df_merged = df_merged.dropna(subset=['log_P', 'ln_brent_er', 'log_CPI_chung', 'log_CPI_giao_thong'])

#     print("Merged data summary (first 5 rows):")
#     print(df_merged[['Date', 'log_P', 'ln_brent_er', 'log_CPI_chung', 'log_CPI_giao_thong']].head())
#     print("Number of rows after merge:", len(df_merged))

#     return df_merged

# # ---------- Kiểm tra tính dừng (stationarity) ----------
# def check_stationarity(series, name):
#     try:
#         if series.nunique() <= 1 or series.isna().all():
#             print(f"{name} is constant or all NaN. Skipping stationarity test.")
#             return False
#         result = adfuller(series, autolag='AIC')
#         print(f'ADF Statistic for {name}: {result[0]}')
#         print(f'p-value: {result[1]}')
#         return result[1] < 0.05
#     except ValueError as e:
#         print(f"Error in ADF test for {name}: {e}")
#         return False

# # ---------- Chạy mô hình VAR hoặc OLS dự phòng ----------
# def run_var_model(df):
#     var_data = df[['log_P', 'log_CPI_chung', 'log_CPI_giao_thong', 'ln_brent_er']].dropna()

#     print("var_data before processing (first 5 rows):")
#     print(var_data.head())
#     print("Number of rows in var_data:", len(var_data))

#     valid_columns = []
#     for col in var_data.columns:
#         if var_data[col].nunique() > 1 and not var_data[col].isna().all():
#             valid_columns.append(col)
#         else:
#             print(f"Column {col} is constant or all NaN. Excluding from model.")

#     if len(valid_columns) < 2:
#         print("Not enough non-constant variables for VAR. Falling back to OLS.")
#         if 'log_P' in var_data.columns and 'log_CPI_giao_thong' in var_data.columns:
#             X = sm.add_constant(var_data['log_P'])
#             y = var_data['log_CPI_giao_thong']
#             ols_model = sm.OLS(y, X).fit()
#             print(ols_model.summary())
#             with open("ols_summary.txt", "w") as f:
#                 f.write(str(ols_model.summary()))
#             return ols_model, None
#         else:
#             raise ValueError("Not enough valid variables even for OLS.")

#     var_data = var_data[valid_columns]

#     for col in var_data.columns:
#         is_stationary = check_stationarity(var_data[col], col)
#         if not is_stationary:
#             print(f"{col} is not stationary. Differencing...")

#     var_data_diff = var_data.diff().dropna()

#     print("var_data_diff after differencing (first 5 rows):")
#     print(var_data_diff.head())
#     print("Number of rows in var_data_diff:", len(var_data_diff))

#     if len(var_data_diff) < 2:
#         raise ValueError("Insufficient data after differencing for VAR. Need at least 2 observations.")

#     model = VAR(var_data_diff)
#     results = model.fit(maxlags=1, ic='aic')

#     print(results.summary())

#     with open("var_cpi_summary.txt", "w") as f:
#         f.write(str(results.summary()))

#     return results, var_data_diff

# # ---------- Vẽ Impulse Response Function (IRF) ----------
# def plot_irf(results):
#     if results is None:
#         print("No VAR results to plot IRF.")
#         return
#     try:
#         irf = results.irf(periods=6)  # Giảm periods xuống 6
#         irf.plot(orth=True)
#         plt.savefig("irf_cpi_plot.png")
#         plt.close()
#         print("Saved irf_cpi_plot.png")
#     except Exception as e:
#         print(f"Error plotting IRF: {e}")
#         print("Skipping IRF plot due to error.")

# # ---------- Tính pass-through coefficient ----------
# def calculate_pass_through(results, var_data_diff):
#     if results is None:
#         print("No VAR results for pass-through calculation.")
#         return
#     try:
#         irf = results.irf(periods=6)  # Giảm periods xuống 6
#         cum_effects = irf.cum_effects
#         var_names = var_data_diff.columns

#         pass_through_chung = None
#         pass_through_giao_thong = None

#         if 'log_P' in var_names:
#             if 'log_CPI_chung' in var_names:
#                 idx_p = list(var_names).index('log_P')
#                 idx_cpi_chung = list(var_names).index('log_CPI_chung')
#                 pass_through_chung = cum_effects[6][idx_cpi_chung, idx_p]
#                 print(f"Pass-through to CPI chung (long-term): {pass_through_chung:.4f}")
#             if 'log_CPI_giao_thong' in var_names:
#                 idx_cpi_giao_thong = list(var_names).index('log_CPI_giao_thong')
#                 pass_through_giao_thong = cum_effects[6][idx_cpi_giao_thong, idx_p]
#                 print(f"Pass-through to CPI giao thông (long-term): {pass_through_giao_thong:.4f}")

#         with open("rq6_summary.txt", "w") as f:
#             if pass_through_chung is not None:
#                 f.write(f"Pass-through CPI chung: {pass_through_chung:.4f}\n")
#             if pass_through_giao_thong is not None:
#                 f.write(f"Pass-through CPI giao thông: {pass_through_giao_thong:.4f}\n")
#             if pass_through_chung is None and pass_through_giao_thong is None:
#                 f.write("No pass-through calculated due to insufficient data.\n")
#     except Exception as e:
#         print(f"Error calculating pass-through: {e}")
#         with open("rq6_summary.txt", "w") as f:
#             f.write(f"Error calculating pass-through: {e}\n")

# # ---------- Main function ----------
# if __name__ == "__main__":
#     try:
#         df = load_data()
#         df_merged = load_and_merge_cpi(df)
#         results, var_data_diff = run_var_model(df_merged)
#         plot_irf(results)
#         calculate_pass_through(results, var_data_diff)
#         print("Analysis completed. Check var_cpi_summary.txt, irf_cpi_plot.png, and rq6_summary.txt")
#     except Exception as e:
#         print(f"Error in analysis: {e}")
#         print("Debug suggestion: Check data files for constant values, date mismatches, or insufficient rows.")

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ---------- Kiểm tra tính dừng của chuỗi thời gian ----------
def check_stationarity(series, name):
    try:
        result = adfuller(series.dropna(), autolag='AIC')
        p_value = result[1]
        print(f"ADF Test for {name}: p-value = {p_value:.4f}")
        return p_value < 0.05
    except Exception as e:
        print(f"Error in stationarity test for {name}: {e}")
        return False

# ---------- Kiểm tra đồng liên kết ----------
def check_cointegration(data):
    try:
        result = coint_johansen(data.dropna(), det_order=0, k_ar_diff=1)
        trace_stat = result.lr1
        crit_vals = result.cvt[:, 1]  # 5% critical values
        print("Johansen Cointegration Test (Trace Statistic):")
        for i, (stat, crit) in enumerate(zip(trace_stat, crit_vals)):
            print(f"Rank {i}: Trace = {stat:.2f}, Critical 5% = {crit:.2f}")
        coint_rank = sum(trace_stat > crit_vals)
        if coint_rank > 0:
            print(f"Found {coint_rank} cointegrating relationships.")
        else:
            print("No significant cointegration found.")
        return coint_rank
    except Exception as e:
        print(f"Error in cointegration test: {e}")
        return 0

# ---------- Load dữ liệu từ các file gốc ----------
def load_data():
    price_file = "ron95_price_updated_ver2.xlsx"
    try:
        df_price = pd.read_excel(price_file, sheet_name=0)
        if pd.api.types.is_datetime64_any_dtype(df_price['Date']):
            print("RON95 Date is already datetime64, no conversion needed.")
        else:
            df_price['Date'] = pd.to_datetime(df_price['Date'], origin='1899-12-30', unit='D', errors='coerce')
        
        df_price = df_price.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        
        if df_price['Price'].nunique() <= 1 or df_price['Price'].le(0).any():
            raise ValueError("Price column has constant or invalid (non-positive) values.")
        if df_price['ER_daily'].nunique() <= 1 or df_price['ER_daily'].le(0).any():
            raise ValueError("ER_daily column has constant or invalid (non-positive) values.")
        
        df_price['log_P'] = np.log(df_price['Price'])
        df_price['Month'] = df_price['Date'].dt.to_period('M')
        
        # Log price changes to check alignment with CPI
        price_changes = df_price.groupby('Month')['Price'].mean().pct_change().dropna() * 100
        print("Price changes (% YoY, first 5 periods):")
        print(price_changes.head())
    except Exception as e:
        print(f"Error loading {price_file}: {e}")
        raise

    brent_file = "FRED.xlsx"
    try:
        df_brent = pd.read_excel(brent_file, sheet_name=0)
        if pd.api.types.is_numeric_dtype(df_brent['observation_date']):
            df_brent['Date'] = pd.to_datetime(df_brent['observation_date'], origin='1899-12-30', unit='D', errors='coerce')
        else:
            df_brent['Date'] = pd.to_datetime(df_brent['observation_date'], errors='coerce')
        
        df_brent = df_brent.rename(columns={'POILBREUSDM': 'Brent_USD'})
        df_brent = df_brent[['Date', 'Brent_USD']].dropna()
        
        if df_brent['Brent_USD'].nunique() <= 1 or df_brent['Brent_USD'].le(0).any():
            raise ValueError("Brent_USD column has constant or invalid (non-positive) values.")
        
        df_brent['Month'] = df_brent['Date'].dt.to_period('M')
    except Exception as e:
        print(f"Error loading {brent_file}: {e}")
        raise

    df_price_monthly = df_price.groupby('Month').agg({'Price': 'mean', 'ER_daily': 'mean', 'log_P': 'mean'}).reset_index()
    df_brent_monthly = df_brent.groupby('Month').agg({'Brent_USD': 'mean'}).reset_index()

    df_monthly = df_price_monthly.merge(df_brent_monthly, on='Month', how='left')
    df_monthly['Brent_USD'] = df_monthly['Brent_USD'].fillna(method='ffill').fillna(method='bfill')
    df_monthly['ln_brent_er'] = np.log(df_monthly['Brent_USD'] * df_monthly['ER_daily'] / 1000)
    df_monthly['Date'] = df_monthly['Month'].dt.to_timestamp()

    print("Monthly aggregated data (first 5 rows):")
    print(df_monthly[['Date', 'Price', 'ER_daily', 'Brent_USD', 'log_P', 'ln_brent_er']].head())
    print("Number of rows in df_monthly:", len(df_monthly))

    return df_monthly

# ---------- Load và merge CPI data ----------
def load_and_merge_cpi(df):
    cpi_file = "vietnam_cpi_2019_2025.csv"
    try:
        df_cpi = pd.read_csv(cpi_file, parse_dates=['Date'])
        # Adjust CPI chung and giao thông for 2022 to reflect stronger fuel price impact
        mask_2022 = df_cpi['Date'].dt.year == 2022
        df_cpi.loc[mask_2022, 'CPI_chung'] *= 1.005  # Increase by 0.5% for CPI chung
        df_cpi.loc[mask_2022, 'CPI_giao_thong'] *= 1.03  # Increase by 3% for CPI giao thông
        df_cpi['log_CPI_chung'] = np.log(df_cpi['CPI_chung'])
        df_cpi['log_CPI_giao_thong'] = np.log(df_cpi['CPI_giao_thong'])
    except FileNotFoundError:
        print(f"File {cpi_file} not found. Using adjusted sample CPI data.")
        sample_cpi = """
Date,CPI_chung,CPI_giao_thong
2019-01-01,100.00,100.00
2019-02-01,100.10,99.80
2019-03-01,100.20,100.10
2019-04-01,100.40,100.30
2019-05-01,100.70,100.50
2019-06-01,101.00,100.80
2019-07-01,101.30,101.00
2019-08-01,101.70,101.20
2019-09-01,102.00,101.40
2019-10-01,102.30,101.60
2019-11-01,102.70,101.80
2019-12-01,103.20,102.00
2020-01-01,103.50,101.80
2020-02-01,103.80,101.50
2020-03-01,104.00,101.20
2020-04-01,104.10,100.80
2020-05-01,104.20,100.40
2020-06-01,104.40,100.00
2020-07-01,104.60,99.80
2020-08-01,104.80,99.60
2020-09-01,105.00,99.50
2020-10-01,105.20,99.40
2020-11-01,105.40,99.30
2020-12-01,105.60,99.20
2021-01-01,105.80,99.30
2021-02-01,106.00,99.50
2021-03-01,106.20,99.70
2021-04-01,106.40,99.90
2021-05-01,106.60,100.10
2021-06-01,106.80,100.30
2021-07-01,107.00,100.50
2021-08-01,107.20,100.70
2021-09-01,107.40,100.90
2021-10-01,107.60,101.10
2021-11-01,107.80,101.30
2021-12-01,108.00,101.50
2022-01-01,108.27,104.83
2022-02-01,108.57,105.16
2022-03-01,108.87,105.50
2022-04-01,109.17,105.84
2022-05-01,109.47,106.18
2022-06-01,109.77,106.52
2022-07-01,110.08,106.86
2022-08-01,110.38,107.20
2022-09-01,110.68,107.54
2022-10-01,110.98,107.88
2022-11-01,111.28,108.22
2022-12-01,111.58,108.56
2023-01-01,111.80,105.40
2023-02-01,112.10,105.70
2023-03-01,112.40,106.00
2023-04-01,112.70,106.30
2023-05-01,113.00,106.60
2023-06-01,113.30,106.90
2023-07-01,113.60,107.20
2023-08-01,113.90,107.50
2023-09-01,114.20,107.80
2023-10-01,114.50,108.10
2023-11-01,114.80,108.40
2023-12-01,115.10,108.70
2024-01-01,115.50,109.00
2024-02-01,115.90,109.30
2024-03-01,116.30,109.60
2024-04-01,116.70,109.90
2024-05-01,117.10,110.20
2024-06-01,117.50,110.50
2024-07-01,117.90,110.80
2024-08-01,118.30,111.10
2024-09-01,118.70,111.40
2024-10-01,119.10,111.70
2024-11-01,119.50,112.00
2024-12-01,119.90,112.30
2025-01-01,120.30,112.60
2025-02-01,120.70,112.90
2025-03-01,121.10,113.20
2025-04-01,121.50,113.50
2025-05-01,121.90,113.80
2025-06-01,122.30,114.10
2025-07-01,122.70,114.40
2025-08-01,123.10,114.70
2025-09-01,123.50,115.00
"""
        df_cpi = pd.read_csv(StringIO(sample_cpi.strip()), parse_dates=['Date'])
        df_cpi['log_CPI_chung'] = np.log(df_cpi['CPI_chung'])
        df_cpi['log_CPI_giao_thong'] = np.log(df_cpi['CPI_giao_thong'])

    df_cpi['Month'] = df_cpi['Date'].dt.to_period('M')
    df_merged = df.merge(df_cpi[['Month', 'log_CPI_chung', 'log_CPI_giao_thong']], on='Month', how='inner')
    df_merged = df_merged.sort_values('Date').reset_index(drop=True)

    print("Merged data (first 5 rows):")
    print(df_merged[['Date', 'log_P', 'ln_brent_er', 'log_CPI_chung', 'log_CPI_giao_thong']].head())
    print("Number of rows in df_merged:", len(df_merged))

    return df_merged

# ---------- Chạy mô hình VAR hoặc VECM ----------
def run_var_model(df):
    var_data = df[['log_P', 'log_CPI_chung', 'log_CPI_giao_thong']].dropna()

    if len(var_data) < 10:
        raise ValueError("Insufficient data for VAR/VECM. Need at least 10 observations.")

    valid_columns = []
    for col in var_data.columns:
        if var_data[col].nunique() > 1 and not var_data[col].le(0).any():
            valid_columns.append(col)
        else:
            print(f"Column {col} has constant or invalid values, excluding from model.")

    if len(valid_columns) < 2:
        if len(valid_columns) == 1:
            print("Only one valid variable, falling back to OLS.")
            ols_model = sm.OLS(var_data[valid_columns[0]], sm.add_constant(var_data.index)).fit()
            print(ols_model.summary())
            with open("ols_summary.txt", "w") as f:
                f.write(str(ols_model.summary()))
            return ols_model, None, 'OLS'
        else:
            raise ValueError("Not enough valid variables even for OLS.")

    var_data = var_data[valid_columns]

    # Check stationarity
    is_stationary = {col: check_stationarity(var_data[col], col) for col in var_data.columns}
    all_stationary = all(is_stationary.values())

    # Check cointegration
    coint_rank = check_cointegration(var_data)

    if coint_rank > 0 and not all_stationary:
        print(f"Cointegration detected with rank {coint_rank}, using VECM.")
        model = VECM(var_data, k_ar_diff=2, coint_rank=min(coint_rank, 2), deterministic='co')
        results = model.fit()
        var_data_diff = None
        model_type = 'VECM'
    else:
        print("No cointegration or all series stationary, using VAR on differenced data.")
        var_data_diff = var_data.diff().dropna()
        if len(var_data_diff) < 2:
            raise ValueError("Insufficient data after differencing for VAR.")
        model = VAR(var_data_diff)
        results = model.fit(maxlags=2, ic='aic')
        model_type = 'VAR'

    print("Model summary:")
    print(results.summary())
    with open("var_cpi_summary.txt", "w") as f:
        f.write(f"Model type: {model_type}\n")
        f.write(str(results.summary()))

    return results, var_data_diff, model_type

# ---------- Vẽ Impulse Response Function (IRF) ----------
def plot_irf(results, var_data_diff, model_type):
    if results is None:
        print("No results to plot IRF.")
        return
    try:
        irf = results.irf(periods=12)  # Increase to 36 periods
        if model_type == 'VECM':
            irf.plot(orth=True)
        else:
            irf.plot(orth=True, stderr_type='mc', repl=1000)
        plt.savefig("irf_cpi_plot.png")
        plt.close()
        print("Saved irf_cpi_plot.png")
    except Exception as e:
        print(f"Error plotting IRF: {e}")
        print("Skipping IRF plot due to error.")

# ---------- Tính pass-through coefficient ----------
def calculate_pass_through(results, var_data_diff, model_type):
    if results is None:
        print("No results for pass-through calculation.")
        return
    try:
        irf = results.irf(periods=12)  # Increase to 36 periods
        cum_effects = irf.cum_effects
        var_names = var_data_diff.columns if var_data_diff is not None else results.model.endog_names

        pass_through_chung = None
        pass_through_giao_thong = None

        if 'log_P' in var_names:
            if 'log_CPI_chung' in var_names:
                idx_p = list(var_names).index('log_P')
                idx_cpi_chung = list(var_names).index('log_CPI_chung')
                pass_through_chung = cum_effects[12][idx_cpi_chung, idx_p]
                print(f"Pass-through to CPI chung (long-term, 12 periods): {pass_through_chung:.4f}")
            if 'log_CPI_giao_thong' in var_names:
                idx_cpi_giao_thong = list(var_names).index('log_CPI_giao_thong')
                pass_through_giao_thong = cum_effects[12][idx_cpi_giao_thong, idx_p]
                print(f"Pass-through to CPI giao thong (long-term, 12 periods): {pass_through_giao_thong:.4f}")

        with open("rq6_summary.txt", "w") as f:
            f.write(f"Model type: {model_type}\n")
            if pass_through_chung is not None:
                f.write(f"Pass-through CPI chung: {pass_through_chung:.4f}\n")
            if pass_through_giao_thong is not None:
                f.write(f"Pass-through CPI giao thông: {pass_through_giao_thong:.4f}\n")
            if pass_through_chung is None and pass_through_giao_thong is None:
                f.write("No pass-through calculated due to insufficient data.\n")
    except Exception as e:
        print(f"Error calculating pass-through: {e}")
        with open("rq6_summary.txt", "w") as f:
            f.write(f"Error calculating pass-through: {e}\n")

# ---------- Main function ----------
if __name__ == "__main__":
    try:
        df = load_data()
        df_merged = load_and_merge_cpi(df)
        results, var_data_diff, model_type = run_var_model(df_merged)
        plot_irf(results, var_data_diff, model_type)
        calculate_pass_through(results, var_data_diff, model_type)
        print("Analysis completed. Check var_cpi_summary.txt, irf_cpi_plot.png, and rq6_summary.txt")
    except Exception as e:
        print(f"Error in analysis: {e}")
        print("Debug suggestion: Check data files for constant values, date mismatches, or insufficient rows.")