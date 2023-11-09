import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# 最小-最大归一化
def min_max_normalization(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data, max_val - min_val, min_val


# 标准化
def standardization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data, std, mean


info = pd.read_csv("./house_info_new.csv")
columns = info.columns

np_district = info['区域'].to_numpy()

np_D = info['室'].to_numpy()
np_D, k, b = min_max_normalization(np_D)

np_L = info['厅'].to_numpy()
np_L, k, b = min_max_normalization(np_L)

np_area = info['房间大小'].to_numpy()
np_area, k, b = min_max_normalization(np_area)

np_east = info['东'].to_numpy()
np_south = info['南'].to_numpy()
np_west = info['西'].to_numpy()
np_north = info['北'].to_numpy()

np_fixture = info['装修造型'].to_numpy()

np_year = info['修建年份'].to_numpy()
np_year, k, b = min_max_normalization(np_year)

np_hType = info['房源'].to_numpy()
np_lift = info['电梯'].to_numpy()
np_env = info['绿化率'].to_numpy()
np_env, k, b = min_max_normalization(np_env)

np_price = info['平米房价'].to_numpy()
np_price, k, b = min_max_normalization(np_price)

info = np.asarray(
    [np_district, np_D, np_L, np_area, np_east, np_south, np_west, np_north, np_fixture, np_year, np_hType, np_lift,
     np_env, np_price]).T
info = pd.DataFrame(np.asarray(info), columns=columns)

print(info)

# 假设area是地区特征的索引，例如0表示区域A，1表示区域B，以此类推
area_indices = torch.tensor([0, 1, 2, 0, 1])

# 定义嵌入层
embedding_dim = 4  # 设置嵌入维度
num_areas = 10  # 假设有10个不同的地区
embedding = nn.Embedding(num_areas, embedding_dim)

# 将地区特征映射到嵌入向量
embedded_areas = embedding(area_indices)

print(embedded_areas)

# data.to_csv("./house_info_new.csv", encoding="utf-8-sig", index=False)
