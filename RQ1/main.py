import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.vector_ar.svar_model import SVAR
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

try:
    from linearmodels.iv import IV2SLS
    has_linearmodels = True
except Exception:
    has_linearmodels = False
    print("linearmodels not installed — script will use manual 2SLS fallback. To install: pip install linearmodels")

# ---------- USER CONFIG ----------
PRICE_FILE = "ron95_price_updated_ver2.xlsx"
BRENT_FILE = "FRED.xlsx"
BVMT_VND = 186120833.0
TTDB_PCT = 0.10
VAT_BREAK_DATE = pd.to_datetime("2025-07-01")
VAT_BEFORE = 0.10
VAT_AFTER = 0.08
# ---------------------------------

# Helper function to parse date
def parse_date_from_df(df):
    for c in df.columns:
        colname = str(c).lower()
        if colname in ["date", "ngày", "day", "ngay_dieu_chinh", "observation_date"]:
            return df, c
    return df, None

# ---------- 1. Read price file ----------
try:
    df_price = pd.read_excel(PRICE_FILE, sheet_name=0)
except Exception as e:
    raise FileNotFoundError(f"Không thể đọc file {PRICE_FILE}: {e}")
if "Ngay_dieu_chinh" in df_price.columns:
    df_price = df_price.rename(columns={"Ngay_dieu_chinh": "Date"})

# Debug: In dữ liệu gốc
print("Dữ liệu gốc df_price (trước parse):\n", df_price.head())
print("Cột trong df_price:", df_price.columns.tolist())

# Parse Date
if pd.api.types.is_numeric_dtype(df_price["Date"]):
    df_price["Date"] = pd.to_datetime(df_price["Date"], origin='1899-12-30', unit='D', errors="coerce")
else:
    df_price["Date"] = pd.to_datetime(df_price["Date"], errors="coerce")
print("Dữ liệu df_price sau parse Date:\n", df_price[["Date"]].head())
print("NaN trong Date sau parse:", df_price["Date"].isna().sum())

df_price = df_price.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
print("Price rows:", len(df_price), "Date range:", df_price["Date"].min(), "-", df_price["Date"].max())

# Debug: Kiểm tra phân bố tháng
df_price["year"] = df_price["Date"].dt.year
df_price["month"] = df_price["Date"].dt.month
print("Phân bố tháng trong df_price:\n", df_price.groupby(["year", "month"]).size())

# Thêm dữ liệu giả cho post-2025-07-01 (bạn có thể bỏ comment nếu cần)
# fake_data = pd.DataFrame({
#     "Date": [pd.to_datetime("2025-07-15"), pd.to_datetime("2025-08-15")],
#     "RetailPrice_VND_lit": [25000, 25500],
#     "Quantity": [1000000, 1050000],
#     "QBOG_balance": [0, 0]
# })
# df_price = pd.concat([df_price, fake_data], ignore_index=True)

df_price, date_col = parse_date_from_df(df_price)
if date_col is None:
    raise ValueError("Could not determine Date column in price file. Ensure a Date column exists or Year+Month. Aborting.")
df_price = df_price.rename(columns={date_col: "Date"})

# Thêm dummy mùa vụ (m_1 đến m_12)
month_dummies = pd.get_dummies(df_price['month'], prefix='m', drop_first=True)
df_price = pd.concat([df_price, month_dummies], axis=1)


# ---------- Tái tạo giá bán lẻ tuyệt đối ----------
absolute_price_cols = [c for c in df_price.columns if "price" in c.lower() or "gia" in c.lower()]
if absolute_price_cols:
    price_col = absolute_price_cols[0]
    df_price[price_col] = df_price[price_col].astype(str).str.replace(' đ', '').str.replace('.', '').str.replace(',', '')
    df_price["RetailPrice_VND_lit"] = pd.to_numeric(df_price[price_col], errors="coerce")
    if df_price["RetailPrice_VND_lit"].dtype == 'object':
        raise ValueError(f"Cột {price_col} sau khi làm sạch vẫn chứa giá trị không phải số. Kiểm tra dữ liệu: {df_price[price_col].head()}")
    print(f"Dùng cột {price_col} làm giá bán lẻ tuyệt đối.")
else:
    raise ValueError("Không tìm thấy cột giá bán lẻ tuyệt đối (Price, RetailPrice_VND_lit, ...). Kiểm tra dữ liệu.")

# Tạo biến log_P và log_Q
df_price["log_P"] = np.log(df_price["RetailPrice_VND_lit"])
df_price["log_Q"] = np.log(df_price["Quantity"])

# ---------- Read Brent file ----------
try:
    df_brent = pd.read_excel(BRENT_FILE, sheet_name=0)
except Exception as e:
    raise FileNotFoundError(f"Không thể đọc file {BRENT_FILE}: {e}")

# Kiểm tra và chuyển đổi observation_date
if pd.api.types.is_numeric_dtype(df_brent["observation_date"]):
    # Nếu là số (Excel serial date), chuyển sang datetime
    df_brent["Date"] = pd.to_datetime(df_brent["observation_date"], origin='1899-12-30', unit='D')
elif pd.api.types.is_datetime64_any_dtype(df_brent["observation_date"]):
    # Nếu đã là datetime, sử dụng trực tiếp
    df_brent["Date"] = pd.to_datetime(df_brent["observation_date"])
else:
    raise ValueError("Cột observation_date phải là số hoặc datetime. Kiểm tra lại file FRED.xlsx.")

df_brent = df_brent.rename(columns={"POILBREUSDM": "Brent_USD"})
print("Dữ liệu df_brent sau parse Date:\n", df_brent[["Date", "Brent_USD"]].head())

# Merge with price data
df = df_price.merge(df_brent[["Date", "Brent_USD"]], on="Date", how="left")

# Sửa NaN trong Brent_USD bằng ffill (điền giá trị trước đó)
df["Brent_USD"] = df["Brent_USD"].fillna(method='ffill').fillna(method='bfill')  # ffill rồi bfill nếu đầu NaN

df["ER_daily"] = df_price["ER_daily"]  # Sử dụng ER_daily từ file mới
df["ln_brent_er"] = np.log(df["Brent_USD"] * df["ER_daily"] / 1000)  # Giả sử Brent_USD * ER_daily / 1000 = VND/lit
df["ln_brent_er_lag1"] = df["ln_brent_er"].shift(1)
# Thêm biến mới
df["Inflation_rate"] = df_price["Inflation_rate"]
df["Holiday_dummy"] = df_price["Holiday_dummy"]

# Thêm dummy cho VAT và chính sách
df['dummy_172025'] = (df['Date'] >= VAT_BREAK_DATE).astype(int)  # VAT 8% từ 01/07/2025
df['VAT_rate'] = np.where(df['Date'] >= VAT_BREAK_DATE, VAT_AFTER, VAT_BEFORE)
df['Policy_t'] = df['VAT_rate'] + TTDB_PCT + (df['QBOG_balance'].abs() / 1e9) + (BVMT_VND / 1e9) + df['Holiday_dummy'] * 0.01
df["QBOG_balance"] = df["QBOG_balance"].fillna(0.0)
df["QBOG_use"] = np.abs(df["QBOG_balance"])
df["QBOG_use_norm"] = df["QBOG_use"] / df["QBOG_use"].mean()

# ---------- Model A: Độ co giãn cầu - 2SLS/3SLS ----------
# Phương trình log-log
if has_linearmodels:
    # 1) Prepare df_iv
    df_iv = df[['log_Q','log_P','Inflation_rate','Holiday_dummy','dummy_172025',
                'ln_brent_er','ln_brent_er_lag1', 'VAT_rate', 'Policy_t', "QBOG_use_norm"]].dropna().copy()

    # df_iv = df[['log_Q','log_P','Inflation_rate','Holiday_dummy','dummy_172025',
    #             'ln_brent_er','ln_brent_er_lag1','ln_brent_er_lag2','m_2','m_3','m_4','m_5','m_6','m_7','m_8','m_9','m_10','m_11']].dropna().copy()
    # print("df_iv.shape:", df_iv.shape)

    # 2) Drop constant columns (no variation)
    const_cols = [c for c in df_iv.columns if df_iv[c].nunique() <= 1]
    if const_cols:
        print("Dropping constant columns:", const_cols)
        df_iv.drop(columns=const_cols, inplace=True)

    # 3) Define variables
    dependent = df_iv['log_Q']                      # biến phụ thuộc
    exog_no_const = df_iv[['Inflation_rate', 'Policy_t', 'Holiday_dummy', 'VAT_rate', "QBOG_use_norm", 'dummy_172025']].copy()
    exog = sm.add_constant(exog_no_const, has_constant='add')  # controls + const
    endog = df_iv[['log_P']]                        # biến nội sinh (price)
    instruments = df_iv[['ln_brent_er_lag1']]      # instruments

    # 4) Rank check + iterative drop to restore full-rank if needed
    colnames = list(exog.columns) + list(endog.columns)
    X = np.hstack([exog.values, endog.values]).astype(float)
    print("Initial matrix shape:", X.shape, "rank:", np.linalg.matrix_rank(X))
    iter_drop = []
    while np.linalg.matrix_rank(X) < X.shape[1]:
        # check very high pairwise correlation first
        corr = pd.DataFrame(X, columns=colnames).corr().abs()
        hi = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().sort_values(ascending=False)
        if not hi.empty and hi.iloc[0] > 0.995:
            pair = hi.index[0]
            # drop the second of the pair (prefer to drop controls rather than const)
            drop_col = pair[1]
            print(f"Dropping highly correlated column {drop_col} (corr={hi.iloc[0]:.3f})")
        else:
            # use SVD: find singular vector of smallest singular value
            U, s, Vt = np.linalg.svd(X)
            smallest_v = Vt[-1]
            idx = int(np.argmax(np.abs(smallest_v)))
            drop_col = colnames[idx]
            print(f"Dropping column {drop_col} detected by SVD (smallest singular vector).")
        iter_drop.append(drop_col)

        # actually drop from exog or endog
        if drop_col in exog.columns:
            exog = exog.drop(columns=[drop_col])
            if drop_col in exog_no_const.columns:
                exog_no_const = exog_no_const.drop(columns=[drop_col])
        elif drop_col in endog.columns:
            endog = endog.drop(columns=[drop_col])
        else:
            print("Unexpected column to drop:", drop_col)
            break

        # rebuild
        colnames = list(exog.columns) + list(endog.columns)
        if len(colnames) == 0:
            raise ValueError("No regressors left after dropping collinear columns.")
        X = np.hstack([exog.values, endog.values])
        print("Updated matrix shape:", X.shape, "rank:", np.linalg.matrix_rank(X))

    if iter_drop:
        print("Automatically dropped columns to restore rank:", iter_drop)

    # 5) Check instrument condition
    if instruments.shape[1] < endog.shape[1]:
        print("ERROR: số instrument < số biến nội sinh. Không thể nhận dạng. instruments:", instruments.columns.tolist(), "endog:", endog.columns.tolist())

    # 6) First-stage check (instrument strength)
    try:
        first_stage_exog = pd.concat([exog_no_const, instruments], axis=1)
        first_stage = sm.OLS(endog, sm.add_constant(first_stage_exog, has_constant='add')).fit()
        print("First-stage summary:\n", first_stage.summary())
        try:
            ftest = first_stage.f_test("ln_brent_er_lag1 = 0")
            print("F-test on instrument ln_brent_er_lag1:", ftest)
        except Exception:
            pass
    except Exception as e:
        print("First-stage cannot be estimated:", e)

    # 7) Try IV2SLS (positional args)
    try:
        iv_model = IV2SLS(dependent, exog, endog, instruments).fit(cov_type='robust')
        print("IV2SLS succeeded. Summary:")
        with open("RQ1_res/manual_2sls_summary.txt", "w") as f:
            f.write(iv_model.summary.as_text())

        elasticity = iv_model.params['log_P']
        p_value = iv_model.pvalues['log_P']
        with open("RQ1_res/elasticity_summary.txt", "w") as f:
            f.write(f"Demand Elasticity (coefficient of log_P): {elasticity:.4f}\n")
            f.write(f"P-value: {p_value:.4f}\n")
            f.write(f"Interpretation: A 1% increase in price leads to a {elasticity:.4f}% change in quantity demanded.\n")
            f.write(f"Expected: Elasticity should be negative (~ -0.1 to -0.5 for gasoline). If positive, check instruments or data.\n")
    except Exception as e_iv:
        print("IV2SLS failed:", e_iv)
        print("Falling back to manual 2SLS (first-stage OLS to get fitted price).")
        try:
            Z = sm.add_constant(pd.concat([exog_no_const, instruments], axis=1), has_constant='add')
            first = sm.OLS(df_iv['log_P'], Z).fit()
            df_iv['log_P_hat'] = first.fittedvalues
            X2 = sm.add_constant(pd.concat([df_iv['log_P_hat'], exog_no_const], axis=1), has_constant='add')
            second = sm.OLS(df_iv['log_Q'], X2).fit(cov_type='HC1')
            print("Manual 2SLS summary:\n", second.summary())
        except Exception as e2:
            print("Manual 2SLS also failed:", e2)

else:
    print("linearmodels not installed — cannot run IV2SLS.")


# Vẽ biểu đồ
plt.figure(figsize=(8, 5))
plt.plot(df["Date"], df["RetailPrice_VND_lit"], marker='o')
plt.title("Price (VND) over time")
plt.xlabel("Date"); plt.ylabel("Price VND"); plt.grid(True)
plt.tight_layout()
plt.savefig("RQ1_res/price_time_series.png")
print("Saved price_time_series.png")

plt.figure(figsize=(6, 6))
if "log_Q" in df.columns and df["log_Q"].notna().sum() > 0:
    plt.scatter(df["log_P"], df["log_Q"], alpha=0.6)
    plt.xlabel("log_P"); plt.ylabel("log_Q"); plt.title("log(Q) vs log(P)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("RQ1_res/scatter_logQ_logP.png")
    print("Saved scatter_logQ_logP.png")
else:
    df["log_P"].hist(bins=30)
    plt.title("Distribution of log_P (no Quantity available)")
    plt.tight_layout()
    plt.savefig("hist_logP.png")
    print("Saved hist_logP.png")

print("Model A finished.")

# ---------- Model B: Đồng tích hợp & pass-through - ARDL/VECM/SVAR ----------
df['dummy_11121'] = (df['Date'] >= pd.to_datetime("2021-11-01")).astype(int) # Giai đoạn dịch COVID-19
df['dummy_24222'] = (df['Date'] >= pd.to_datetime("2022-02-24")).astype(int)  # Xung đột Nga - Ukraine
df['dummy_010723'] = (df['Date'] >= pd.to_datetime("2023-07-01")).astype(int)  # Khôi phục thuế môi trường
df['dummy_010124'] = (df['Date'] >= pd.to_datetime("2024-01-01")).astype(int)  # Năm tài chính 2024
df['dummy_112025'] = (df['Date'] >= pd.to_datetime("2025-01-07")).astype(int) # Giảm VAT 10% -> 8%

# Vector Z_t với các biến mới
df['ln_ER_daily'] = np.log(df['ER_daily'] / 1000)  # Giả sử ER_daily / 1000 để đồng nhất đơn vị
df['Tax_t'] = df['VAT_rate'] + TTDB_PCT + (BVMT_VND / 1e9)
df['CPI_trans'] = df['Inflation_rate']  # Giả sử CPI_trans là proxy của Inflation_rate

# Debug Z_t
print("df shape:", df.shape)
print("NaN in df:\n", df.isnull().sum())

# Điền NaN cho các cột chính để tăng số quan sát
df['ln_brent_er'] = df['ln_brent_er'].fillna(method='ffill').fillna(method='bfill')  # Điền ln_brent_er
df['ln_ER_daily'] = df['ln_ER_daily'].fillna(method='ffill').fillna(method='bfill')  # Điền ln_ER_daily
df['Tax_t'] = df['Tax_t'].fillna(method='ffill')  # Điền Tax_t

Z_t = df[['log_P', 'ln_brent_er', 'ln_ER_daily', 'QBOG_balance', 'Tax_t', 'log_Q', 'Inflation_rate', 'Holiday_dummy', 'dummy_172025', 'dummy_112025']].dropna()

# Debug sau sửa
print("Z_t shape after dropna:", Z_t.shape)
print("NaN in Z_t:\n", Z_t.isnull().sum())

# Giảm exog để giảm regressors (chỉ giữ 4 biến chính để n=230 > regressors ~9)
exog_reduced = Z_t[['ln_brent_er', 'ln_ER_daily', 'Inflation_rate', 'dummy_172025']]

# ARDL model với exog giảm và order=0 để giảm lag nếu cần
try:
    ardl = ARDL(Z_t['log_P'], lags=1, exog=exog_reduced, order=1)
    ardl_res = ardl.fit()
    print("ARDL results:")
    with open("RQ1_res/ardl_summary.txt", "w") as f:
        f.write(ardl_res.summary().as_text())
    
    # Sửa phần tính độ co giãn dài hạn cho RQ1: Sử dụng cách tiếp cận IV thủ công cho ARDL để xử lý nội sinh
    # First stage: ARDL cho log_P sử dụng instruments và controls
    # Chỉ sử dụng các cột có trong Z_t, lấy VAT_rate, Policy_t, QBOG_use_norm, ln_brent_er_lag1 từ df
    exog_controls = Z_t[['Inflation_rate', 'Holiday_dummy', 'dummy_172025']].copy()
    # Lấy VAT_rate, Policy_t, QBOG_use_norm, ln_brent_er_lag1 từ df và đồng bộ index
    extra_controls = df[['VAT_rate', 'Policy_t', 'QBOG_use_norm', 'Inflation_rate', 'Holiday_dummy','dummy_172025', 'dummy_112025','Tax_t', 'ln_brent_er_lag1']].reindex(Z_t.index)
    exog_controls = pd.concat([exog_controls, extra_controls], axis=1).dropna()
    
    # Instrument từ extra_controls
    instruments = extra_controls[['ln_brent_er_lag1']].reindex(Z_t.index).dropna()
    
    # Kết hợp exog cho first stage
    exog_first = pd.concat([exog_controls, instruments], axis=1).dropna()
    
    # Đồng bộ Z_t với exog_first
    common_index = exog_first.index.intersection(Z_t.index)
    Z_t = Z_t.loc[common_index].copy()
    exog_first = exog_first.loc[common_index]
    instruments = instruments.loc[common_index]
    
    ardl_first = ARDL(Z_t['log_P'], lags=1, exog=exog_first, order=1)
    ardl_first_res = ardl_first.fit()
    Z_t['log_P_hat'] = ardl_first_res.fittedvalues
    
    # Second stage: ARDL cho log_Q sử dụng log_P_hat và controls (không instrument)
    exog_second = pd.concat([exog_controls, Z_t[['log_P_hat']]], axis=1).dropna()
    
    # Đồng bộ lại nếu cần
    common_index = exog_second.index.intersection(Z_t.index)
    Z_t = Z_t.loc[common_index].copy()
    exog_second = exog_second.loc[common_index]
    
    ardl_demand = ARDL(Z_t['log_Q'], lags=1, exog=exog_second, order=1)
    ardl_demand_res = ardl_demand.fit()
    print("ARDL Demand model for long-run elasticity (with IV approach):")
    with open("RQ1_res/ardl_demand_summary.txt", "w") as f:
        f.write(ardl_demand_res.summary().as_text())
        print("Saved ardl_demand_summary.txt")
    
    # Tính ngắn hạn: hệ số L0 của log_P_hat
    short_term_elasticity = ardl_demand_res.params.get('log_P_hat.L0', ardl_demand_res.params.get('log_P_hat', 0))
    
    # Tính dài hạn: tổng hệ số log_P_hat / (1 - hệ số autoregressive của log_Q)
    ar_coef = ardl_demand_res.params.get('log_Q.L1', 0)
    long_term_elasticity = (short_term_elasticity + ardl_demand_res.params.get('log_P_hat.L1', 0)) / (1 - ar_coef)
    
    print(f"Short-term elasticity: {short_term_elasticity:.4f}")
    print(f"Long-term elasticity: {long_term_elasticity:.4f}")
    
    with open("RQ1_res/elasticity_summary.txt", "a") as f:  # Append để thêm vào file gốc
        f.write(f"\nLong-term Demand Elasticity: {long_term_elasticity:.4f}\n")
except ValueError as e:
    print("ARDL failed:", e)
    print("Thử giảm order=0...")
    ardl = ARDL(Z_t['log_P'], lags=1, exog=exog_reduced, order=0)
    ardl_res = ardl.fit()
    print("ARDL with order=0 results:")
    with open("ardl_summary.txt", "w") as f:
        f.write(ardl_res.summary().as_text())

# ---------- Tính tổn thất phúc lợi (DWL) nếu giá công bố < giá cân bằng ----------
# Giả định giá cân bằng là giá không ràng buộc (từ mô hình OLS dự đoán giá không dummy ràng buộc)
# Giả định hàm cầu tuyến tính cho đơn giản: DWL = 1/2 * (P_cân bằng - P_trần) * (Q_cân bằng - Q_thực)
# Sử dụng elasticity từ RQ1, giả định P_trần = giá công bố, P_cân bằng = giá dự đoán từ OLS không ràng buộc, Q_thực = Quantity thực, Q_cân bằng = Q_thực * (1 + elasticity * (P_cân bằng - P_trần)/P_trần)

# Mô hình OLS để dự đoán giá cân bằng (không dummy ràng buộc)
ols_equilibrium = sm.OLS(df['log_P'], sm.add_constant(df[['ln_brent_er', 'ln_ER_daily', 'Inflation_rate']])).fit()
df['log_P_equilibrium'] = ols_equilibrium.predict()
df['P_equilibrium'] = np.exp(df['log_P_equilibrium'])

# Giả định ràng buộc khi dummy_172025 = 1 (hoặc điều kiện khác, giả sử dummy_172025 là ràng buộc)
df['P_published'] = df['RetailPrice_VND_lit']  # Giá công bố (thực tế)
df['Q_real'] = df['Quantity']  # Lượng thực

# Lấy elasticity ngắn hạn từ file (hoặc từ code trước)
elasticity = -0.1196  # Từ elasticity_summary.txt, thay bằng giá trị âm nếu điều chỉnh dữ liệu

# Tính Q_cân bằng = Q_real * (1 + elasticity * % thay đổi giá)
df['delta_P_pct'] = (df['P_equilibrium'] - df['P_published']) / df['P_published']
df['Q_equilibrium'] = df['Q_real'] * (1 + elasticity * df['delta_P_pct'])

# Tính DWL chỉ khi P_published < P_equilibrium (giá trần)
mask = (df['P_published'] < df['P_equilibrium'])
df['DWL'] = 0
df.loc[mask, 'DWL'] = 0.5 * (df.loc[mask, 'P_equilibrium'] - df.loc[mask, 'P_published']) * (df.loc[mask, 'Q_equilibrium'] - df.loc[mask, 'Q_real'])

# Tổng DWL
total_dwl = df['DWL'].sum()
print(f"Tổng tổn thất phúc lợi (DWL): {total_dwl:.2f} VND")

# Lưu vào file
with open("RQ1_res/dwl_summary.txt", "w") as f:
    f.write(f"(DWL): {total_dwl:.2f} VND\n")
    f.write("DWL based on: 1/2 * deltaP * deltaQ\n")