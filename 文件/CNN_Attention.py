import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 假设输入数据是一维的，形状为 (batch_size, 300)，这里将其视为单通道一维信号
        # 为了使用卷积层，需要将其reshape为 (batch_size, 1, 300)
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.pool3 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(512 * 110, 256)  # 37是经过三次MaxPool1d(2)后300维度的剩余长度（300 -> 150 -> 75 -> 37）
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.output = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.46)
        self.attention = Attention(64, 64)

    def forward(self, x):
        # 将输入reshape为 (batch_size, 1, 300) 以适应Conv1d输入要求
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        # 将卷积层输出展平以输入全连接层
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn6(self.fc3(x)))
        x = self.dropout(x)
        x = self.attention(x)
        x = torch.sigmoid(self.output(x))
        return x

