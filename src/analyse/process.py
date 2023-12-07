"""
处理院房屋数据house_info.csv
处理后的数据将被保存在Processed.csv
"""
import pandas
import pandas as pd
import numpy as np
import torch.nn as nn
import torch

DataFrame = pd.read_csv("../static/house_info.csv")


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


def process(data: pandas.DataFrame):
    district = data['district'].to_numpy()
    region = data['region'].to_numpy()
    buildingYear = data['buildingYear'].to_numpy()
    housePlan = data['housePlan'].to_numpy()
    houseArea = data['houseArea'].to_numpy()
    houseType = data['houseType'].to_numpy()
    buildingType = data['buildingType'].to_numpy()
    # houseComprisingArea = data['houseComprisingArea'].to_numpy()
    houseOrientation = data['houseOrientation'].to_numpy()
    buildingStructure = data['buildingStructure'].to_numpy()
    houseDecoration = data['houseDecoration'].to_numpy()
    price = data['price'].to_numpy()
    longitude = data['longitude'].to_numpy()
    latitude = data['latitude'].to_numpy()
    print(district, region, buildingYear, housePlan, houseArea, houseType, buildingType, houseType,
          houseOrientation, buildingStructure, houseDecoration, longitude, latitude, price, sep="\n")


if __name__ == "__main__":
    process(DataFrame)
    pass
