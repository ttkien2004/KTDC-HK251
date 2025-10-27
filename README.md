# 🔍 Dự án Phân tích Kinh tế Lượng – Giá Xăng Dầu Việt Nam

## 📘 Tổng quan

Dự án này thực hiện **phân tích mối quan hệ giữa giá bán lẻ xăng dầu, giá dầu Brent, tỷ giá, thuế và CPI**, dựa trên dữ liệu thực tế từ các nguồn:
- `ron95_price_updated_ver1/2.xlsx`: dữ liệu giá bán lẻ RON95 theo thời gian
- `FRED.xlsx`: dữ liệu giá dầu Brent (USD/thùng)
- Các biến vĩ mô: tỷ giá (`ER_daily`), lạm phát (`Inflation_rate`), thuế VAT và thuế bảo vệ môi trường (BVMT)

Mục tiêu của dự án:
- **RQ1**: Ước lượng **độ co giãn cầu xăng dầu** bằng mô hình **2SLS (Instrumental Variables)**  
- **RQ2–RQ4**: Phân tích **cơ chế truyền dẫn giá** và **tác động của chính sách thuế (VAT, BVMT)** đến giá bán lẻ  
- **RQ6**: Phân tích **tác động của chỉ số giá tiêu dùng (CPI)** đến giá xăng dầu

---

## ⚙️ Thiết lập môi trường

### 1️⃣ Cài đặt Pipenv (nếu chưa có)

```bash
pip install pipenv
```

### 2️⃣ Cài môi trường ảo cho toàn project

Tại thư mục gốc"

```bash
pipenv install
```

Lệnh này sẽ tự động cài các gói:

- pandas, numpy
- statsmodels, linearmodels
- matplotlib, openpyxl

Sau đó kích hoạt môi trường:

```bash
pipenv shell
```

## Chạy từng phần

### RQ1 – Mô hình 2SLS (Độ co giãn cầu)

Chạy lệnh

```bash
cd RQ1
pipenv run python main.py
```

Kết quả đầu ra:

- RQ1_res/manual_2sls_summary.txt – kết quả mô hình 2SLS

- RQ1_res/elasticity_summary.txt – độ co giãn cầu

- RQ1_res/dwl_summary.txt – tổn thất phúc lợi (DWL)

- Biểu đồ price_time_series.png, scatter_logQ_logP.png
  
### RQ2 - 4

**RQ2 - Cơ chế truyền dẫn giá (ARDL)**

```bash
cd RQ2-4
pipenv run python pass_through_statistics.py
```

Kết quả đầu ra

- RQ2/ardl_results_with_tax_dummy.txt – kết quả ARDL và độ trễ truyền dẫn

**RQ4 - Tác động của VAT và BVMT**

```bash
cd RQ2-4
pipenv run python vat_statistics.py
```

Kết quả đầu ra

- RQ4/rq4_summary_improved.txt – kết quả ARDL/VECM và tính tổn thất phúc lợi

- RQ4/price_after_tax_plot.png – biểu đồ giá sau thuế

**RQ6 - Tác động của CPI đến giá xăng dầu**

Chạy lệnh

```bash
cd RQ6
pipenv run python cpi_analysis.py
```

Kết quả đầu ra

- rq6_summary.txt

- var_cpi_summary.txt

- irf_cpi_plot.png

## Tổng kết

| Mục tiêu    | Phương pháp          | File chính                   | Đầu ra                    |
| ----------- | -------------------- | ---------------------------- | ------------------------- |
| **RQ1**     | 2SLS / IV Regression | `main.py`                    | Elasticity, DWL           |
| **RQ2**     | ARDL                 | `pass_through_statistics.py` | Pass-through coefficients |
| **RQ3–RQ4** | ARDL / VECM          | `vat_statistics.py`          | VAT/BVMT effects          |
| **RQ6**     | OLS / ARDL           | `cpi_analysis.py`            | CPI impact                |

## Ghi chú

Dự án yêu cầu Python 3.10+

**Tác giả**: Nhóm 07 - Lớp L01

**Cập nhật**: Tháng 10/2025
