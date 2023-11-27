import pandas as pd
import re


def getDataFrame():
    df = pd.read_csv("./house_info.csv", encoding="utf-8-sig")
    district = df['district'].to_numpy()
    region = df['region'].to_numpy()
    df['buildingYear'] = df['buildingYear'].apply(lambda x: applyRe(x, "(\\d+)"))
    buildingYear = df['buildingYear'].to_numpy()
    housePlan = df['housePlan'].to_numpy()
    df['houseArea'] = df['houseArea'].apply(lambda x: applyRe(x, "(\\d+)"))
    houseArea = df['houseArea'].to_numpy()
    df['houseType'] = df['houseType'].apply(lambda x: setNone(x, "暂无数据"))
    houseType = df['houseType'].to_numpy()
    df['buildingType'] = df['buildingType'].apply(lambda x: setNone(x, "暂无数据"))
    buildingType = df['buildingType'].to_numpy()
    df['houseComprisingArea'] = df['houseComprisingArea'].apply(lambda x: applyRe(x,"(\\d+)"))
    houseComprisingArea = df['houseComprisingArea'].to_numpy()
    houseOrientation = df['houseOrientation'].to_numpy()
    df['buildingStructure'] = df['buildingStructure'].apply(lambda x: setNone(x, "未知结构"))
    buildingStructure = df['buildingStructure'].to_numpy()
    df['houseDecoration'] = df['houseDecoration'].apply(lambda x: setNone(x, "其他"))
    houseDecoration = df['houseDecoration'].to_numpy()
    price = df['price'].to_numpy()
    longitude = df['longitude'].to_numpy()
    latitude = df['latitude'].to_numpy()
    return df


def applyRe(x: str, pattern: str):
    try:
        return re.search(pattern, x).group(1)
    except AttributeError:
        return None


def setNone(x: str, pattern: str):
    if x == pattern:
        return None
    else:
        return x


if __name__ == "__main__":
    getDataFrame().to_csv("Processed.csv", encoding="utf-8-sig", index=False)
