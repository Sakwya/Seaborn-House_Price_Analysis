import os

import numpy as np
import torch
import torch.nn as nn
from core import dataHander as dh


def embedding(data: np.ndarray):
    """
    对数据进行嵌入
    :param data:
    :return:
    """
    dataList = np.unique(data).tolist()
    dataList.sort()
    embedding_dim = int(len(dataList) ** 0.5) + 1
    embedding = nn.Embedding(len(dataList), embedding_dim)
    for i in range(len(data)):
        data[i] = dataList.index(data[i])
    data = torch.tensor(data.tolist())
    embedded_data = embedding(data)
    return embedded_data, embedding


def onehot(data: np.ndarray):
    """
    对数据进行独热编码
    :param data:
    :return:
    """
    data = data.tolist()
    dataList = set(())
    for i in range(len(data)):

        data[i] = data[i].split()
        if '' in data[i]:
            data[i].remove('')
        for data_label in data[i]:
            if data_label not in dataList:
                dataList.add(data_label)
    dataList = list(dataList)
    dataList.sort()
    onehot_data = np.zeros((len(data), len(dataList)))
    for i in range(len(data)):
        data_label = data[i]
        for _ in data_label:
            onehot_data[[i], [dataList.index(_)]] = 1
    onehot_data = torch.tensor(onehot_data.tolist())
    return onehot_data, dataList


def normalization(data: np.ndarray):
    """
    对数据进行归一化
    :param data:
    :return:
    """
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    normalized_data = torch.tensor(normalized_data.tolist())
    normalized_data.unsqueeze(1)
    return normalized_data.view(len(normalized_data), 1), max_val - min_val, min_val


def standardization(data: np.ndarray):
    """
    对数据进行标准化
    :param data:
    :return:
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    standardized_data = torch.tensor(standardized_data.tolist())
    standardized_data.unsqueeze(1)
    return standardized_data.view(len(standardized_data), 1), std, mean


def other_process(data: np.ndarray):
    """
    对数据进行其他处理
    :param data:
    :return:
    """
    data = torch.tensor(data.tolist())
    return data.view(len(data), 1)


class DataProcess:
    """
    数据处理

    数据一共含有以下信息：

    - District: 采用嵌入
    - HouseType: 采用独热
    - Orientation: 采用独热
    - Fixture: 采用嵌入
    - Bedroom: 采用归一化
    - LivingRoom: 采用归一化
    - Area: 采用归一化
    - Year: 采用归一化
    - Lift: 不用特殊处理
    - GreenRate: 采用标准化
    - Price: 采用归一化
    """

    def __init__(self, from_csv: bool = True, from_pt: bool = False):
        if from_csv:
            self.dataFrame = dh.getDataFrame()
            self.columns = ['District', 'HouseType', 'Orientation', 'Fixture', 'Bedroom', 'LivingRoom', 'Area', 'Year',
                            'Lift', 'GreenRate', 'Price']
            self.district = self.dataFrame["District"].to_numpy()
            self.houseType = self.dataFrame["HouseType"].to_numpy()
            self.orientation = self.dataFrame["Orientation"].to_numpy()
            self.fixture = self.dataFrame["Fixture"].to_numpy()
            self.bedroom = self.dataFrame["Bedroom"].to_numpy()
            self.livingRoom = self.dataFrame["LivingRoom"].to_numpy()
            self.area = self.dataFrame["Area"].to_numpy()
            self.year = self.dataFrame["Year"].to_numpy()
            self.lift = self.dataFrame["Lift"].to_numpy()
            self.greenRate = self.dataFrame["GreenRate"].to_numpy()
            self.price = self.dataFrame["Price"].to_numpy()

            # 嵌入
            self.embedding_district = embedding(self.district)
            self.embedding_fixture = embedding(self.fixture)
            # 独热
            self.onehot_houseType = onehot(self.houseType)
            self.onehot_orientation = onehot(self.orientation)
            # 归一化
            self.normalization_bedroom = normalization(self.bedroom)
            self.normalization_livingRoom = normalization(self.livingRoom)
            self.normalization_area = normalization(self.area)
            self.normalization_year = normalization(self.year)
            self.normalization_price = normalization(self.price)
            # 标准化
            self.standardization_greenRate = standardization(self.greenRate)
            # 其它
            self.other_process_lift = other_process(self.lift)
            self.data = torch.cat(
                (self.embedding_district[0],
                 self.embedding_fixture[0],
                 self.onehot_houseType[0],
                 self.onehot_orientation[0],
                 self.normalization_bedroom[0],
                 self.normalization_livingRoom[0],
                 self.normalization_area[0],
                 self.normalization_year[0],
                 self.standardization_greenRate[0],
                 self.other_process_lift,
                 ), dim=1
            )
            self.target = self.normalization_price[0]
            return
        if from_pt:
            if not os.path.exists('model/data.pt'):
                raise FileNotFoundError("model/data.pt is missed.")
            if not os.path.exists('model/target.pt'):
                raise FileNotFoundError("model/target.pt is missed.")
            self.data = torch.load('model/data.pt')
            self.target = torch.load('model/target.pt')

    def save_data(self):
        if not os.path.exists('model'):
            os.mkdir('model')
        torch.save(self.data, 'model/data.pt')
        torch.save(self.target, 'model/target.pt')


if __name__ == "__main__":
    dp = DataProcess()
    dp.save_data()
