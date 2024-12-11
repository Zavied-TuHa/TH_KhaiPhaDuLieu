import numpy as np
from sklearn.cluster import KMeans

# Tập dữ liệu
data = np.array([5, 10, 11, 13, 15, 35, 50, 55, 72, 92, 204, 215])

# (a) Equal-Frequency
n_bins = 3
equal_frequency_bins = np.array_split(np.sort(data), n_bins)

# (b) Equal-Width
min_val, max_val = np.min(data), np.max(data)
width = (max_val - min_val) / n_bins
equal_width_bins = [data[(data >= min_val + i * width) & (data < min_val + (i + 1) * width)]
                    for i in range(n_bins)]
equal_width_bins[-1] = np.append(equal_width_bins[-1], data[data == max_val])

# (c) Clustering
kmeans = KMeans(n_clusters=n_bins, random_state=0, n_init=10)
clusters = kmeans.fit_predict(data.reshape(-1, 1))
cluster_bins = [data[clusters == i] for i in range(n_bins)]

# In kết quả
print("Phân chia theo tần suất bằng nhau (Equal-Frequency):")
for i, bin_ in enumerate(equal_frequency_bins, 1):
    print(f"Ngăn {i}: {bin_}")

print("\nPhân chia theo chiều rộng bằng nhau (Equal-Width):")
for i, bin_ in enumerate(equal_width_bins, 1):
    print(f"Ngăn {i}: {bin_}")

print("\nPhân cụm (Clustering):")
for i, bin_ in enumerate(cluster_bins, 1):
    print(f"Cụm {i}: {bin_}")
