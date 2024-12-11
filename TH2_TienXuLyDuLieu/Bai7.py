import numpy as np

# Tập dữ liệu
data = np.array([200, 300, 400, 600, 1000])

# (a) Chuẩn hóa Min-Max
min_val, max_val = np.min(data), np.max(data)
min_max_normalized = (data - min_val) / (max_val - min_val)

# (b) Chuẩn hóa Z-Score
mean = np.mean(data)
std_dev = np.std(data)
z_score_normalized = (data - mean) / std_dev

# (c) Chuẩn hóa Z-Score bằng Mean Absolute Deviation (MAD)
mad = np.mean(np.abs(data - mean))
mad_normalized = (data - mean) / mad

# (d) Chuẩn hóa theo Tỉ lệ Thập Phân (Decimal Scaling)
j = np.ceil(np.log10(np.max(np.abs(data))))
decimal_scaled = data / (10**j)

print("(a) Chuẩn hóa Min-Max:")
print(min_max_normalized)

print("\n(b) Chuẩn hóa Z-Score:")
print(z_score_normalized)

print("\n(c) Chuẩn hóa Z-Score bằng MAD:")
print(mad_normalized)

print("\n(d) Chuẩn hóa theo Tỉ lệ Thập Phân:")
print(decimal_scaled)
