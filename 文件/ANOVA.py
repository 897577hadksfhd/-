import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

def FeatureReducer():
    positive_train_data = pd.read_csv('../480/kcr_cv+480.csv')
    negative_train_data = pd.read_csv('../480/kcr_cv-480.csv')
    positive_test_data = pd.read_csv('GUI/pos_esm_features.csv')
    negative_test_data = pd.read_csv('GUI/neg_esm_features.csv')

    positive_train_data['label'] = 1
    negative_train_data['label'] = 0
    positive_test_data['label'] = 1
    negative_test_data['label'] = 0

    # 合并
    train_data = pd.concat([positive_train_data, negative_train_data], axis=0)
    test_data = pd.concat([positive_test_data, negative_test_data], axis=0)

    # 提取特征矩阵和标签
    X_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values
    X_test = test_data.drop('label', axis=1).values
    y_test = test_data['label'].values

    # 使用方差分析进行特征选择
    selector = SelectKBest(score_func=f_classif, k=300)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # 重新划分正、负数据
    positive_train_selected = X_train_selected[:len(positive_train_data)]
    negative_train_selected = X_train_selected[len(positive_train_data):]
    positive_test_selected = X_test_selected[:len(positive_test_data)]
    negative_test_selected = X_test_selected[len(positive_test_data):]

    # 将选择后的正、负数据转换为 DataFrame 并保存
    positive_train_df = pd.DataFrame(positive_train_selected)
    negative_train_df = pd.DataFrame(negative_train_selected)
    positive_test_df = pd.DataFrame(positive_test_selected)
    negative_test_df = pd.DataFrame(negative_test_selected)

    positive_train_df.to_csv('GUI/train_anova+.csv', index=False)
    negative_train_df.to_csv('GUI/train_anova-.csv', index=False)
    positive_test_df.to_csv('GUI/pos_esm_ANOVA.csv', index=False)
    negative_test_df.to_csv('GUI/neg_esm_ANOVA.csv', index=False)

    print("前 300 维特征已选择并按正负数据保存到相应文件中。")

    return positive_test_df, negative_test_df





