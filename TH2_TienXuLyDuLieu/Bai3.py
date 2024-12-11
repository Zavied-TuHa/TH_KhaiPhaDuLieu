import numpy as np

# Data
point1 = np.array([22, 1, 42, 10])
point2 = np.array([20, 0, 36, 8])

# a. Tính khoảng cách Euclid
euclidean_distance = np.linalg.norm(point1 - point2)

# b. Tính khoảng cách Manhattan
manhattan_distance = np.sum(np.abs(point1 - point2))

# c. Tính khoảng cách Minkowski với q = 3
q = 3
minkowski_distance = np.sum(np.abs(point1 - point2) ** q) ** (1 / q)

# d. Tính khoảng cách cực đại
max_distance = np.max(np.abs(point1 - point2))

print(f"Khoảng cách Euclid: {euclidean_distance:.2f}")
print(f"Khoảng cách Manhattan: {manhattan_distance:.2f}")
print(f"Khoảng cách Minkowski (q=3): {minkowski_distance:.2f}")
print(f"Khoảng cách cực đại: {max_distance:.2f}")