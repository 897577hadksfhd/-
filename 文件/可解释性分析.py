import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import shap
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
import numpy as np


# 加载模型（保持和之前相同的模型结构）
class Attention(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(in_features, hidden_dim)
        self.key = nn.Linear(in_features, hidden_dim)
        self.value = nn.Linear(in_features, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(query.shape[-1]).float())
        attention_weights = self.softmax(attention_scores)
        attention_output = torch.matmul(attention_weights, value)
        return attention_output


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(880, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.output = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.46)
        self.attention = Attention(64, 64)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.attention(x)
        x = torch.sigmoid(self.output(x))
        return x


# 加载训练好的模型
model = MLP()
model.load_state_dict(torch.load('DNN_ATT-esm+anova1.pth'))
model.eval()


# SHAP模型包装器
class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            tensor_X = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(tensor_X)
            return outputs.detach().numpy()


# 分开读取正负数据（假设数据中没有标签列）
positive_data = pd.read_csv('../onehot+esm2/onehot+anova/test-onehot+anova+.csv')
negative_data = pd.read_csv('../onehot+esm2/onehot+anova/test-onehot+anova-.csv')

# 合并数据（直接合并，因为没有标签列）
all_data = pd.concat([positive_data, negative_data])

# 转换为numpy数组
X_all = all_data.values
X_positive = positive_data.values
X_negative = negative_data.values

# 创建背景数据（从合并数据中随机采样）
background = X_all[np.random.choice(X_all.shape[0], 10, replace=False)]

# 创建解释器
explainer = shap.Explainer(ModelWrapper(model).predict, background)

# 计算SHAP值（限制样本数量以提高速度）
sample_size = min(1000, len(X_all))  # 最多1000个样本
shap_values_all = explainer.shap_values(X_all[:sample_size])
#shap_values_positive = explainer.shap_values(X_positive[:sample_size // 2])
#shap_values_negative = explainer.shap_values(X_negative[:sample_size // 2])

# 创建特征名称（880维）
feature_names = [f'feat_{i}' for i in range(880)]

# 可视化结果
plt.figure(figsize=(15, 15))

# 1. 所有数据的SHAP摘要图
#plt.subplot(1, 3, 1)
shap.summary_plot(shap_values_all, X_all[:sample_size], feature_names=feature_names, show=False)
plt.title("All Samples")

# 2. 正样本的SHAP摘要图
# plt.subplot(1, 3, 2)
# shap.summary_plot(shap_values_positive, X_positive[:sample_size // 2], feature_names=feature_names, show=False)
# plt.title("Positive Samples")
#
# # 3. 负样本的SHAP摘要图
# plt.subplot(1, 3, 3)
# shap.summary_plot(shap_values_negative, X_negative[:sample_size // 2], feature_names=feature_names, show=False)
# plt.title("Negative Samples")

plt.tight_layout()
plt.show()

# # 可选：分析特定特征（例如前5个特征）
# for i in range(5):
#     shap.dependence_plot(f'feat_{i}', shap_values_all, X_all[:sample_size],
#                          feature_names=feature_names,
#                          title=f'Feature {i} Dependence')