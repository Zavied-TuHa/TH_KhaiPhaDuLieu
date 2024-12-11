import pandas as pd
from scipy.io import arff
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Đọc dữ liệu từ file ARFF
data, meta = arff.loadarff(r'D:\Code\Python\KPDL\TH3\vote.arff')
df = pd.DataFrame(data)

# Chuyển đổi byte sang string (nếu cần)
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.decode('utf-8')

# One-hot encoding cho dữ liệu
df_encoded = pd.get_dummies(df)

#Thông số của Apriori
lowerBoundMinSupport = 0.05
minMetric = 0.9
numRules = 10
verbose = False

# Khai phá luật kết hợp
frequent_itemsets = apriori(df_encoded, min_support=lowerBoundMinSupport, use_colnames=True)

#Tính số lượng itemsets
num_itemsets = len(frequent_itemsets)

# Tạo các luật kết hợp
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minMetric, num_itemsets = num_itemsets)

#Giới hạn số lượng luật kết hợp
rules = rules.head(numRules)

# In kết quả
print("Apriori")
print("=======")
print(f"Minimum support: {lowerBoundMinSupport} ({len(df) * lowerBoundMinSupport:.0f} instances)")
print(f"Minimum metric <confidence>: {minMetric}")

print("Generated sets of large itemsets:")
for i in range(1, 5):
    itemsets_size = len(frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == i)])
    if itemsets_size > 0:
        print(f"Size of set of large itemsets L({i}): {itemsets_size}")
        
print("\nBest rules found:")
for index, rule in rules.iterrows():
    condition = ' '.join(list(rule['antecedents']))
    conclusion = ' '.join(list(rule['consequents']))
    support_conclusion = df_encoded[conclusion].mean()
    conviction = (1 - support_conclusion) / (1 - rule['confidence']) if rule['confidence'] < 1 else 0
    print(f"{index+1}. {condition} ==> {conclusion} {int(rule['support']*len(df))}    "
          f"<conf:({rule['confidence']:.2f})> lift:({rule['lift']:.2f}) lev:({rule['leverage']:.2f}) "
          f"[{int(rule['support']*len(df))}] conv:({conviction:.2f})")