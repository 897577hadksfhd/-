import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Step 1: 加载数据
positive_data = pd.read_csv('onehot+esm2/onehot+anova/train-onehot+anova+.csv')
negative_data = pd.read_csv('onehot+esm2/onehot+anova/train-onehot+anova-.csv')

# Step 2: 标记数据集
positive_data['label'] = 1  # 正样本
negative_data['label'] = 0  # 负样本

# Step 3: 合并数据集
combined_data = pd.concat([positive_data, negative_data])

# 分析可能的分层特征
print("数据集中的分类特征：")
categorical_features = []
for col in combined_data.columns:
    if combined_data[col].dtype == 'object' or combined_data[col].nunique() < 20:
        categorical_features.append(col)
        print(f"- {col}: {combined_data[col].nunique()} 个不同值")

# 计算各特征与标签的相关性，并选择最佳特征
print("\n特征与标签的相关性：")
best_feature = None
best_chi2 = 0

for feature in categorical_features:
    if feature != 'label':
        # 使用卡方检验计算相关性
        contingency_table = pd.crosstab(combined_data[feature], combined_data['label'])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        print(f"- {feature}: 卡方值 = {chi2:.2f}, p值 = {p:.4f}")

        # 选择卡方值最大的特征
        if chi2 > best_chi2:
            best_chi2 = chi2
            best_feature = feature

if best_feature:
    print(f"\n自动选择最佳分层特征: {best_feature} (卡方值 = {best_chi2:.2f})")
    stratify_feature = best_feature
else:
    print("错误：未找到合适的分层特征。")
    exit(1)

# Step 5: 计算各层的样本分布
strata = combined_data[stratify_feature].unique()
strata_distribution = {}
for stratum in strata:
    stratum_data = combined_data[combined_data[stratify_feature] == stratum]
    positive_count = len(stratum_data[stratum_data['label'] == 1])
    negative_count = len(stratum_data[stratum_data['label'] == 0])
    strata_distribution[stratum] = {'positive': positive_count, 'negative': negative_count}

# Step 6: 确定每层的目标样本数（取最小类别数）
target_samples_per_stratum = {}
for stratum, counts in strata_distribution.items():
    target_samples_per_stratum[stratum] = min(counts['positive'], counts['negative'])

# Step 7: 执行分层下采样
balanced_strata = []
for stratum in strata:
    stratum_data = combined_data[combined_data[stratify_feature] == stratum]
    positive_samples = stratum_data[stratum_data['label'] == 1]
    negative_samples = stratum_data[stratum_data['label'] == 0]

    # 对多数类进行下采样
    target_size = target_samples_per_stratum[stratum]
    if len(positive_samples) > target_size:
        positive_samples = positive_samples.sample(n=target_size, random_state=42)
    if len(negative_samples) > target_size:
        negative_samples = negative_samples.sample(n=target_size, random_state=42)

    # 合并下采样后的样本
    balanced_stratum = pd.concat([positive_samples, negative_samples])
    balanced_strata.append(balanced_stratum)

# Step 8: 合并所有层的平衡数据
balanced_data = pd.concat(balanced_strata)

# Step 9: 分离正负样本
balanced_positive_data = balanced_data[balanced_data['label'] == 1].drop(columns=['label'])
balanced_negative_data = balanced_data[balanced_data['label'] == 0].drop(columns=['label'])

# Step 10: 保存结果
balanced_positive_data.to_csv('onehot+esm2/onehot+anova/onehot+anova+SS2.csv', index=False)
balanced_negative_data.to_csv('onehot+esm2/onehot+anova/onehot+anova-SS2.csv', index=False)

# 打印结果
print("分层下采样结果：")
for stratum, counts in strata_distribution.items():
    original_pos = counts['positive']
    original_neg = counts['negative']
    target = target_samples_per_stratum[stratum]
    print(f"层 '{stratum}': 正样本 {original_pos} -> {target}, 负样本 {original_neg} -> {target}")

print("\n最终平衡数据集:")
print("正样本数量:", len(balanced_positive_data))
print("负样本数量:", len(balanced_negative_data))