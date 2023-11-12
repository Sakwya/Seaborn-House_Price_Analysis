import numpy as np
import pandas as pd


def removeMissData():
    data = getDataFrame()
    print(data)
    data.dropna(axis=0, inplace=True)
    print(data)
    saveDataFrame(data)


def getDataFrame() -> pd.DataFrame:
    """
    读取数据
    :return: 数据 type:DataFrame
    """
    return pd.read_csv("data/house_info.csv", encoding="utf-8-sig", na_values=['NaN', '', ' ', 'NA', 'N/A'])


def saveDataFrame(data: pd.DataFrame, path: str = "data/house_info.csv") -> None:
    """
    存储数据
    :param data: 要存储的数据：
        如果不改变存储路径，需要保证data的列顺序和columns的列顺序一致
        columns = [District, HouseType, Orientation, Fixture, Bedroom, LivingRoom, Area, Year, Lift, GreenRate, Price]
    :param path: 存储的路径，默认覆盖data/house_info.csv
    :type data: pd.DataFrame
    :type path: str
    """
    columns = pd.Index(
        data=['District', 'HouseType', 'Orientation', 'Fixture', 'Bedroom', 'LivingRoom', 'Area', 'Year', 'Lift',
              'GreenRate', 'Price'])
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a DataFrame")
    if path == "data/house_info.csv" and not columns.equals(data.columns):
        raise ValueError("DataFrame columns must be equal to original columns. Or you can  specify a new path.")
    data.to_csv(path_or_buf=path, encoding="utf-8-sig", index=False)


# data = getDataFrame()
# data["Year"] = data["Year"].str.extract('(\d+)').astype(int)
# print(data)
# saveDataFrame(data)

# data = getDataFrame()
# lift = data["Lift"].copy()
# for i in range(len(lift)):
#     if lift[i] == "有":
#         lift[i] = 1
#     else:
#         lift[i] = 0
# data["Lift"] = lift
# print(data)
# saveDataFrame(data)

