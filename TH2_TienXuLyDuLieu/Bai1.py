import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats

# Dữ liệu
data = [13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70]

# a. Tính mean, median, mode
mean = np.mean(data)
median = np.median(data)

# Đếm tần suất xuất hiện của từng giá trị
freq = Counter(data)
max_count = max(freq.values())

# Lấy tất cả các giá trị có tần suất bằng max_count
modes = [key for key, value in freq.items() if value == max_count]

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Mode(s): {modes} (Xuất hiện {max_count} lần)")

# b. Nhận xét về mode
print(f"Nhận xét: Giá trị mode của tập dữ liệu là {modes}. Đây là dữ liệu đơn modal (một mode duy nhất).")

# c. Tính giá trị Midrange
midrange = (min(data) + max(data)) / 2
print(f"Midrange: {midrange}")

# d. Tính tứ phân vị Q1 và Q3
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
print(f"Tứ phân vị đầu tiên (Q1): {q1}")
print(f"Tứ phân vị thứ ba (Q3): {q3}")

# e. Five-number summary
min_val = min(data)
max_val = max(data)
q2 = median
five_number_summary = (min_val, q1, q2, q3, max_val)
print(f"Five-number summary: {five_number_summary}")

# f. Vẽ biểu đồ hộp (Boxplot)
plt.figure(figsize=(10, 6))
box = plt.boxplot(data, vert=False, patch_artist=True, widths=0.6)

for patch in box['boxes']:
    patch.set(facecolor='#DDEBF7')

plt.text(mean, 1.1, f"Mean: {mean:.2f}", color="blue", fontsize=10, ha='center')
plt.text(median, 1.2, f"Median: {median:.2f}", color="green", fontsize=10, ha='center')
modes = [25, 35]
plt.text(14, 1.2, f"Modes: {', '.join(map(str, modes))}", color="red", fontsize=10, ha='center')
plt.text(q1, 0.85, f"Q1: {q1:.2f}", color="purple", fontsize=10, ha='center')
plt.text(q3, 0.85, f"Q3: {q3:.2f}", color="purple", fontsize=10, ha='center')
plt.text(min_val, 0.7, f"Min: {min_val}", color="red", fontsize=10, ha='center')
plt.text(max_val, 0.7, f"Max: {max_val}", color="red", fontsize=10, ha='center')

plt.title("Boxplot của tập dữ liệu", fontsize=14)
plt.xlabel("Giá trị", fontsize=12)
plt.yticks([])
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.show()