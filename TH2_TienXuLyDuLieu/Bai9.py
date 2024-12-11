import numpy as np
import pandas as pd

# Dữ liệu từ bảng
data = {
    "A1": [1.5, 2.0, 1.6, 1.2, 1.5],
    "A2": [1.7, 1.9, 1.8, 1.5, 1.0]
}

# Chuyển đổi thành DataFrame
df = pd.DataFrame(data)

# a. Tính chuẩn hóa z-score cho từng thuộc tính
z_score = df.apply(lambda x: (x - x.mean()) / x.std())

# b. Hệ số tương quan Pearson
pearson_corr = df["A1"].corr(df["A2"])

# c. Hiệp phương sai
covariance = df["A1"].cov(df["A2"])

# In kết quả
print("Z-Score Normalization:")
print(z_score)
print("\nPearson Correlation Coefficient:", pearson_corr)
print("Covariance:", covariance)
