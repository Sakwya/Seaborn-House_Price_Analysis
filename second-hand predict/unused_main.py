import sys
import pandas as pd
import pymysql
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

sys.path.append("..")

# 定义list
one_hot_region_name = []
one_hot_windows_dir = []
one_hot_direction_info = []
one_hot_floor_info = []
one_hot_build_time = []
one_hot_room_resource = []
one_hot_add_tag_info = []
one_hot_tag_info = []

community_name = []  # 小区名
region_name = []  # 区域名
room_num = []  # 室的个数
hall_num = []  # 厅的个数
room_area = []  # 房间大小
windows_dir = []  # 窗户朝向
decoration_info = []  # 装修造型
floor_info = []  # 户层
build_time = []  # 装修年份
room_resource = []  # 房源
add_tag_info = []  # 附加标签
tag_info = []  # 标签
house_price = []  # 房价
house_price_per_squaremeter = []  # 平米房价

# 连接到MySQL数据库
db = pymysql.connect(host='localhost', user='root', password='root', port=3306, db='1')

# 创建游标对象
db.begin()
cs1 = db.cursor()
sql = "SELECT 小区名,区域,室,厅,房间大小,窗户朝向,装修造型,户层,修建年份,房源,附加标签,标签,房价,平米房价 FROM house_info;"
cs1.execute(sql)
alldata = cs1.fetchall()

# 处理类别特征的one-hot编码和数字特征
encoded_features = []
labels = []
for s in alldata:
    community_name.append([s[0]])
    one_hot_region_name.append(s[1])
    region_name.append([s[1]])
    room_num.append([s[2]])
    hall_num.append([s[3]])
    room_area.append([s[4]])
    one_hot_windows_dir.append(s[5])
    windows_dir.append([s[5]])
    one_hot_direction_info.append(s[6])
    decoration_info.append([s[6]])
    one_hot_floor_info.append(s[7])
    floor_info.append([s[7]])
    one_hot_build_time.append(s[8])
    build_time.append([s[8]])
    one_hot_room_resource.append(s[9])
    room_resource.append([s[9]])
    one_hot_add_tag_info.append(s[10])
    add_tag_info.append([s[10]])
    one_hot_tag_info.append(s[11])
    tag_info.append([s[11]])
    house_price.append([s[12]])
    house_price_per_squaremeter.append([s[13]])


    # print(community_name)
    one_hot_region_name = list(set(one_hot_region_name))
    one_hot_windows_dir = list(set(one_hot_windows_dir))
    one_hot_direction_info = list(set(one_hot_direction_info))
    one_hot_floor_info = list(set(one_hot_floor_info))
    one_hot_build_time = list(set(one_hot_build_time))
    one_hot_room_resource = list(set(one_hot_room_resource))
    one_hot_add_tag_info = list(set(one_hot_add_tag_info))
    one_hot_tag_info = list(set(one_hot_tag_info))

    # 数字特征
    numeric_features = [s[2], s[3], s[4]]

    # 类别特征的one-hot编码
    region_idx = one_hot_region_name.index(s[1])
    windows_dir_idx = one_hot_windows_dir.index(s[5])
    direction_info_idx = one_hot_direction_info.index(s[6])
    floor_info_idx = one_hot_floor_info.index(s[7])
    build_time_idx = one_hot_build_time.index(s[8])
    room_resource_idx = one_hot_room_resource.index(s[9])
    add_tag_info_idx = one_hot_add_tag_info.index(s[10])
    tag_info_idx = one_hot_tag_info.index(s[11])

    one_hot_encoded_features = [
        region_idx, windows_dir_idx, direction_info_idx, floor_info_idx,
        build_time_idx, room_resource_idx, add_tag_info_idx, tag_info_idx
    ]

    # 将数字特征和编码后的类别特征合并为一个特征向量
    feature = numeric_features + one_hot_encoded_features

    # 将特征向量添加到encoded_features列表
    encoded_features.append(feature)

    # 将标签添加到labels列表
    labels.append(s[12])  # 房价作为标签（假设s[12]是房价）

# 将encoded_features和labels转换为PyTorch张量
X_train_tensor = torch.tensor(encoded_features, dtype=torch.float32)
y_train_tensor = torch.tensor(labels, dtype=torch.float32)


# 定义神经网络模型
class HousePricePredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HousePricePredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 初始化模型参数
input_size = 11  # 输入特征的维度
hidden_size = 128
output_size = 1
model = HousePricePredictionModel(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失函数，适用于回归问题
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 训练模型
num_epochs = 10000  # 设置训练轮数
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor.view(-1, 1))  # 计算损失

    # 反向传播和优化
    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重

    # 打印训练信息
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 获取模型的权重和偏置参数
for name, param in model.named_parameters():
    print(f'Parameter name: {name}, Size: {param.size()}')
    print(f'Parameter values: {param.data}')

# 使用模型进行预测（假设有一个新的数据样本new_data，将其转换为特征向量，然后使用模型进行预测）

# 假设待预测的新数据包含以下特征：
# 室的个数、厅的个数、房间大小、区域、窗户朝向、装修造型、户层、修建年份、房源、附加标签、标签

# 示例新数据点，特征的顺序应与训练时一致
new_data = [3, 2, 120, ['真如'], ["南"], ["精装"], ["中楼层(共12层)"], ["2001年建"], ["板楼"], [None],
            ["VR看装修 房本满五年 "]]

# 对字符串特征进行处理
# 处理区域特征
region_encoding = region_name.index(new_data[3])  # 假设region_name是一个保存所有区域的列表
# 处理窗户朝向特征（假设one_hot_windows_dir是窗户朝向的所有可能取值）
windows_dir_one_hot = windows_dir.index(new_data[4])

# 处理装修造型特征（假设decoration_styles是装修造型的所有可能取值）
decoration_encoding = decoration_info.index(new_data[5])
# 处理户层特征（假设floor_levels是户层的所有可能取值）
floor_encoding = floor_info.index(new_data[6])
# 处理修建年份特征（假设build_years是修建年份的所有可能取值）
build_year_encoding = build_time.index(new_data[7])
# 处理房源特征（假设room_sources是房源的所有可能取值）
room_source_encoding = room_resource.index(new_data[8])
# 处理附加标签特征（假设additional_tags是附加标签的所有可能取值）
add_tag_encoding = add_tag_info.index(new_data[9])
# 处理标签特征（假设tags是标签的所有可能取值）
tag_encoding = tag_info.index(new_data[10])

# 最终的特征向量，包括数字特征和处理后的类别特征
processed_new_data = [new_data[0], new_data[1], new_data[2], region_encoding, windows_dir_one_hot, decoration_encoding,
                      floor_encoding, build_year_encoding, room_source_encoding, add_tag_encoding, tag_encoding]
print(processed_new_data)
# 将processed_new_data转换为PyTorch张量
X_new_data = torch.tensor(processed_new_data, dtype=torch.float32)

# 确保X_new_data的维度是 (1x47)
if X_new_data.dim() == 1:
    X_new_data = X_new_data.view(1, -1)  # 如果是一维的，将其转换为二维的 (1x特征数)

# 使用模型进行预测
with torch.no_grad():
    model.eval()  # 切换模型为评估模式
    predicted_price = model(X_new_data)
    print(predicted_price)
    print("预测的房价：", predicted_price.item())  # 输出单一的预测值
