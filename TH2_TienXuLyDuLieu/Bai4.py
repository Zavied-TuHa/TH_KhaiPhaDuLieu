import numpy as np
from numpy.linalg import norm

# Tập dữ liệu
data = np.array([
    [1.5, 1.7],
    [2.0, 1.9],
    [1.6, 1.8],
    [1.2, 1.5],
    [1.5, 1.0]
])

# Điểm dữ liệu mới
new_point = np.array([1.4, 1.6])

# (a) Tính các khoảng cách
euclidean_distances = np.linalg.norm(data - new_point, axis=1)
manhattan_distances = np.sum(np.abs(data - new_point), axis=1)
max_distances = np.max(np.abs(data - new_point), axis=1)
cosine_similarities = np.dot(data, new_point) / (norm(data, axis=1) * norm(new_point))

# Xếp hạng mức độ tương đồng
euclidean_ranking = np.argsort(euclidean_distances)
manhattan_ranking = np.argsort(manhattan_distances)
max_ranking = np.argsort(max_distances)
cosine_ranking = np.argsort(-cosine_similarities)

# (b) Chuẩn hóa dữ liệu
normalized_data = data / norm(data, axis=1, keepdims=True)
normalized_new_point = new_point / norm(new_point)
normalized_euclidean_distances = np.linalg.norm(normalized_data - normalized_new_point, axis=1)
normalized_euclidean_ranking = np.argsort(normalized_euclidean_distances)

# In kết quả
print("Khoảng cách Euclid:", euclidean_distances)
print("Khoảng cách Manhattan:", manhattan_distances)
print("Khoảng cách cực đại:", max_distances)
print("Độ tương đồng cosin:", cosine_similarities)

print("\nXếp hạng theo Euclid:", euclidean_ranking + 1)
print("Xếp hạng theo Manhattan:", manhattan_ranking + 1)
print("Xếp hạng theo Cực đại:", max_ranking + 1)
print("Xếp hạng theo Cosin:", cosine_ranking + 1)

print("\nKhoảng cách Euclid trên dữ liệu chuẩn hóa:", normalized_euclidean_distances)
print("Xếp hạng theo Euclid (chuẩn hóa):", normalized_euclidean_ranking + 1)