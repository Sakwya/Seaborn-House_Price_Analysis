import sys
import pandas as pd
import pymysql
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

sys.path.append("..")

#  3 4 5 13 14 17 18 19为非str
#  13为test
row_column = [1, 2, 6, 7, 8, 9, 10, 11, 12, 15, 16, 20]
not_int_row_column = [3, 4, 5, 13, 14, 17, 18, 19]  # 室 厅 房间大小 房价 平米房价 户数 栋数 绿化率
title_column = ['小区名', '区域', '窗户朝向', '装修造型', '户层', '修建年份', '房源', '附加标签', '标签', '交通',
                '电梯', '物业费']
dummy_data = {}
not_int_dummy_data = {}


def readdb():
    db = pymysql.connect(host='localhost', user='root', password='root', port=3306, db='1')
    cs1 = db.cursor()
    sql = "SELECT * FROM house_info"
    cs1.execute(sql)
    print(cs1.fetchall())
    return cs1, db

for i in range(8):
    cs1, db = readdb()
    topic_data = [row[not_int_row_column[i]] for row in cs1.fetchall()]
    not_int_dummy_data[i] = np.array(topic_data).reshape(-1, 1)
    #print(not_int_dummy_data[i].shape)

for i in range(12):
    cs1, db = readdb()
    topic_data = [row[row_column[i]] for row in cs1.fetchall()]
    df = pd.DataFrame({title_column[i]: topic_data})
    dummy_data[i] = pd.get_dummies(df)
    #print(dummy_data[i].shape)









'''# 定义输入数据和目标标签的形状
input_shape = (36651, 19)
output_shape = (36651, 1)

# 创建一个神经网络模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_shape[1], 128)  # 输入层到第一层全连接层
        self.fc2 = nn.Linear(128, 64)  # 第一层到第二层全连接层
        self.fc3 = nn.Linear(64, output_shape[1])  # 第二层到输出层全连接层
        self.relu = nn.ReLU()  # ReLU激活函数

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

model = MLP()

# 定义损失函数和优化器
loss_fn = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 准备数据，将其转换为PyTorch张量
# 这里需要根据您的数据结构进行调整
X_train = torch.tensor(dummy_data[1], dtype=torch.float32)
y_train = torch.tensor(not_int_dummy_data[1], dtype=torch.float32)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 对模型进行评估
# 这里需要根据您的数据结构进行调整
X_test = torch.tensor(dummy_data[1], dtype=torch.float32)
y_test = torch.tensor(not_int_dummy_data[1], dtype=torch.float32)
y_pred = model(X_test)
loss = loss_fn(y_pred, y_test)
print(f'Test Loss: {loss.item():.4f}')
'''
