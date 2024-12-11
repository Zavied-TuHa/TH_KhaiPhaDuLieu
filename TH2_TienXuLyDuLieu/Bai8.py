import numpy as np

# Tập dữ liệu
data = np.array([13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70])

# (a)SBBM
bin_size = 3
bins = [data[i:i + bin_size] for i in range(0, len(data), bin_size)]
smoothed_data = [float(round(np.mean(bin), 2)) for bin in bins]

#smoothed_result = [value for mean in smoothed_data for value in [mean] * bin_size]

print("(a) Dữ liệu làm mịn bin trung bình:")
print(smoothed_data)

#print(smoothed_result)

# (b) Chuẩn hóa Min-Max
min_val, max_val = np.min(data), np.max(data)
min_max_normalized = (35 - min_val) / (max_val - min_val)
print(f"\n(b) Chuẩn hóa Min-Max cho giá trị 35: {min_max_normalized:.2f}")

# (c) Chuẩn hóa Z-Score
mean = np.mean(data)
std_dev = 12.94
z_score_normalized = (35 - mean) / std_dev
print(f"\n(c) Chuẩn hóa Z-Score cho giá trị 35: {z_score_normalized:.2f}")

# (d) Chuẩn hóa theo Tỉ lệ Thập Phân
j = np.ceil(np.log10(np.max(data)))
decimal_scaled = 35 / (10**j)
print(f"\n(d) Chuẩn hóa theo Tỉ lệ Thập Phân cho giá trị 35: {decimal_scaled:.2f}")
