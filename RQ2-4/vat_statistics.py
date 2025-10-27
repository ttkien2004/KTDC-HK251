import pandas as pd
import numpy as np
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import select_order, VECM, coint_johansen
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file ron95_price_updated_ver1.xlsx
df_price = pd.read_excel("ron95_price_updated_ver1.xlsx", sheet_name=0)

# Chuyển đổi cột Date từ số serial Excel sang datetime
if pd.api.types.is_numeric_dtype(df_price["Date"]):
    df_price["Date"] = pd.to_datetime(df_price["Date"], origin='1899-12-30', unit='D', errors="coerce")
else:
    df_price["Date"] = pd.to_datetime(df_price["Date"], errors="coerce")
df_price = df_price.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

# Xử lý cột Price
df_price["RetailPrice_VND_lit"] = pd.to_numeric(df_price["Price"], errors="coerce")
df_price = df_price.dropna(subset=["RetailPrice_VND_lit"])

# Tạo các biến mới
df_price["log_P"] = np.log(df_price["RetailPrice_VND_lit"])
df_price["VAT_rate"] = np.where(df_price["Date"] >= pd.to_datetime("2025-07-01"), 0.08, 0.10)
df_price["BVMT_2025"] = np.where(df_price["Date"] >= pd.to_datetime("2025-01-01"), 1861.20833, 0)
df_price["Price_after_tax"] = df_price["RetailPrice_VND_lit"] * (1 + df_price["VAT_rate"]) + df_price["BVMT_2025"]
df_price["log_P_after_tax"] = np.log(df_price["Price_after_tax"])
df_price["log_Q"] = np.log(df_price["Quantity"])

# Đặt Date làm index và ép tần số hàng ngày
df_price = df_price.set_index("Date")
df_price = df_price.asfreq('D', method='ffill')  # Ép tần số hàng ngày, điền giá trị thiếu bằng forward fill

# Chuẩn bị dữ liệu cho ARDL/VECM
data = df_price[["log_P_after_tax", "log_P", "VAT_rate", "BVMT_2025", "ER_daily", "Inflation_rate"]].dropna()

# Bước 1: Kiểm tra tính dừng của các biến (ADF test)
def adf_test(series, title=''):
    result = adfuller(series, autolag='AIC')
    print(f'ADF Test for {title}:')
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print(f'Critical Values: {result[4]}')
    print(f'Stationary: {"Yes" if result[1] < 0.05 else "No"}\n')

for col in data.columns:
    adf_test(data[col], col)

# Bước 2: Sai phân bậc 1 nếu không dừng
data_diff = data.diff().dropna()
for col in data_diff.columns:
    adf_test(data_diff[col], f'{col} (diff)')

# Bước 3: Kiểm tra đồng liên kết bằng Johansen test
joh_test = coint_johansen(data, det_order=0, k_ar_diff=1)
print("Johansen Cointegration Test:")
print(f'Eigenvalues: {joh_test.eig}')
print(f'Trace statistic: {joh_test.lr1}')
print(f'Critical values (trace): {joh_test.cvt}')
print(f'Cointegration rank: {np.sum(joh_test.lr1 > joh_test.cvt[:, 1])}\n')

# Bước 4: Nếu có đồng liên kết, sử dụng VECM; nếu không, ARDL
coint_rank = np.sum(joh_test.lr1 > joh_test.cvt[:, 1])
if coint_rank > 0:
    print("Sử dụng VECM vì có đồng liên kết.")
    vecm_order = select_order(data, maxlags=10, deterministic="ci")
    vecm = VECM(data, k_ar_diff=vecm_order.aic, coint_rank=coint_rank, deterministic="ci")
    vecm_results = vecm.fit()
    print(vecm_results.summary())
    
    # Tác động dài hạn từ beta (cointegration relations)
    long_run_effects = vecm_results.beta
else:
    print("Sử dụng ARDL vì không có đồng liên kết rõ ràng.")
    # Chọn lags tối ưu cho ARDL
    model = ARDL(data["log_P_after_tax"], 
                 exog=data[["log_P", "VAT_rate", "BVMT_2025", "ER_daily", "Inflation_rate"]],
                 lags=2,  # Tự động chọn lags tối ưu
                 trend="c")
    results = model.fit(use_t=True)  # Sử dụng t-statistic cho độ tin cậy
    print(results.summary())
    
    # Tìm ec param cho tác động dài hạn
    ec_param = next((p for p in results.params.index if p.startswith("ec")), None)
    if ec_param:
        long_run_effects = -results.params[results.params.index.str.startswith("exog")] / results.params[ec_param]
    else:
        long_run_effects = "Not available"

# Ghi kết quả vào file
with open("RQ4/rq4_summary_improved.txt", "w") as f:
    if coint_rank > 0:
        f.write("VECM Model Results:\n")
        f.write(str(vecm_results.summary()) + "\n\n")
        f.write("Long-term Effects (beta coefficients):\n")
        f.write(str(long_run_effects) + "\n")
    else:
        f.write("ARDL Model Results:\n")
        f.write(results.summary().as_text() + "\n\n")
        f.write("Short-term Effects:\n")
        for var, coef in results.params[results.params.index.str.startswith("exog")].items():
            f.write(f"{var}: {coef:.4f}\n")
        f.write("\nLong-term Effects:\n")
        f.write(str(long_run_effects) + "\n")
    
    # Tính tổn thất phúc lợi
    df_price["Price_increase"] = df_price["Price_after_tax"] - df_price["RetailPrice_VND_lit"]
    df_price["Consumer_welfare_loss"] = df_price["Quantity"] * df_price["Price_increase"]
    total_welfare_loss = df_price["Consumer_welfare_loss"].sum()
    f.write(f"\nTotal Consumer Welfare Loss: {total_welfare_loss:.2f} VND\n")

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.plot(df_price.index, df_price["Price_after_tax"], label="Price after tax (VND/lit)")
plt.axvline(pd.to_datetime("2025-07-01"), color='r', linestyle='--', label="VAT 8% effective")
plt.axvline(pd.to_datetime("2025-01-01"), color='g', linestyle='--', label="BVMT 2025 effective")
plt.title("Price after Tax with VAT and BVMT Adjustments")
plt.xlabel("Date")
plt.ylabel("Price (VND/lit)")
plt.legend()
plt.savefig("RQ4/price_after_tax_plot.png")
plt.close()