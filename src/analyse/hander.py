import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
sns.set_palette(sns.hls_palette(n_colors=14))
plt.rcParams['font.sans-serif'] = 'DengXian'


def getDataFrame():
    df = pd.read_csv("../static/house_info.csv", encoding="utf-8-sig")
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
    df['houseComprisingArea'] = df['houseComprisingArea'].apply(lambda x: applyRe(x, "(\\d+)"))
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


def normalization(data: np.ndarray):
    """
    对数据进行归一化
    :param data:
    :return:
    """
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data, max_val - min_val, min_val


def standardization(data: np.ndarray):
    """
    对数据进行标准化
    :param data:
    :return:
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data, std, mean


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


def kMeans(df, size):
    kmeans = KMeans(n_clusters=size)
    kmeans.fit(df)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    plt.figure()
    plt.scatter(x['longitude'], x['latitude'], s=1, c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='r')
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.title('K-means Clustering')
    return plt


def dBSCAN(df, epsilon, min_samples):
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    dbscan.fit(df)
    labels = dbscan.labels_
    plt.figure()
    plt.scatter(x['longitude'], x['latitude'], s=1, c=labels)
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.title('DBSCAN Clustering')
    return plt


def aGglomerative(df, n_clusters):
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    agglomerative.fit(df)
    labels = agglomerative.labels_
    plt.figure()
    plt.scatter(x['longitude'], x['latitude'], s=1, c=labels)
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.title('Agglomerative Clustering')


if __name__ == "__main__":
    # getDataFrame().to_csv("../static/Processed.csv", encoding="utf-8-sig", index=False)
    x = getDataFrame()[['longitude', 'latitude', 'price']]
    x['price'] = normalization(x['price'].to_numpy())[0]
    # kMeans(x, 200)
    aGglomerative(x, 100)

    x['price'] = standardization(x['price'].to_numpy())[0]
    # kMeans(x, 200)
    aGglomerative(x, 100)
    plt.show()
    print(x)
