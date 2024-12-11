import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

# 1. Dữ liệu đầu vào
age = [23, 23, 27, 27, 39, 41, 47, 49, 50, 52, 54, 54, 56, 57, 58, 58, 60, 61]
fat_percentage = [9.5, 26.5, 7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2, 34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2, 35.7]

# 2. Tính các giá trị thống kê cơ bản
age_mean = np.mean(age)
age_median = np.median(age)
age_std = np.std(age)

fat_mean = np.mean(fat_percentage)
fat_median = np.median(fat_percentage)
fat_std = np.std(fat_percentage)

print("Statistics for Age:")
print(f"Mean: {age_mean:.2f}, Median: {age_median}, Standard Deviation: {age_std:.2f}")
print("\nStatistics for Body Fat Percentage:")
print(f"Mean: {fat_mean:.2f}, Median: {fat_median}, Standard Deviation: {fat_std:.2f}")

# 3. Tạo DataFrame để dễ xử lý với pandas
df = pd.DataFrame({'Age': age, 'Fat_Percentage': fat_percentage})

# 4. Vẽ biểu đồ hộp (boxplot)
plt.figure(figsize=(14, 6))

# Biểu đồ hộp cho tuổi
plt.subplot(1, 2, 1)
sns.boxplot(y=df['Age'], color='skyblue')
plt.title('Boxplot of Age')

plt.text(0.1, 55, f'Mean: {age_mean:.2f}\nMedian: {age_median}\nStd: {age_std:.2f}', 
         fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

# Biểu đồ hộp cho phần trăm mỡ cơ thể
plt.subplot(1, 2, 2)
sns.boxplot(y=df['Fat_Percentage'], color='lightgreen')
plt.title('Boxplot of Body Fat Percentage')
plt.text(0.1, 40, f'Mean: {fat_mean:.2f}\nMedian: {fat_median}\nStd: {fat_std:.2f}', 
         fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

plt.tight_layout()
plt.show()

# 5. Vẽ biểu đồ phân tán (scatter plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='Fat_Percentage', data=df, color='orange')
plt.title('Scatter Plot of Age vs. Fat Percentage')
plt.xlabel('Age')
plt.ylabel('Body Fat Percentage')

plt.text(23, 42, f'Age Mean: {age_mean:.2f}\nFat Mean: {fat_mean:.2f}', fontsize=12,
         bbox=dict(facecolor='white', alpha=0.6))
plt.show()

# 6. Vẽ biểu đồ Q-Q plot
plt.figure(figsize=(14, 6))

# Q-Q plot cho tuổi
plt.subplot(1, 2, 1)
stats.probplot(df['Age'], dist="norm", plot=plt)
plt.title('Q-Q Plot for Age')

# Q-Q plot cho phần trăm mỡ cơ thể
plt.subplot(1, 2, 2)
stats.probplot(df['Fat_Percentage'], dist="norm", plot=plt)
plt.title('Q-Q Plot for Body Fat Percentage')

plt.tight_layout()
plt.show()