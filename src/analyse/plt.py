import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import seaborn as sns
import pandas as pd
import base64
from io import BytesIO

df = pd.read_csv("../static/Processed.csv", encoding="utf-8-sig")
sns.set_style("darkgrid")
sns.set_palette(sns.hls_palette(n_colors=14))
plt.rcParams['font.sans-serif'] = 'DengXian'

district_name = [
    "浦东",
    "闵行",
    "宝山",
    "徐汇",
    "普陀",
    "杨浦",
    "长宁",
    "松江",
    "嘉定",
    "黄浦",
    "静安",
    "虹口",
    "青浦",
    "奉贤",
    "金山",
    "崇明"
]


def toBase64():
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close()
    return image_base64


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


def hist_price_district(district):
    if district not in district_name:
        raise ValueError("请输入正确的区名")
    df_district = df[df["district"] == district]
    df_district = df_district["price"]
    total_count = len(df_district)
    ax = sns.histplot(df_district, kde=True)
    plt.title(f"{district}区的房屋价格直方图")
    plt.xlabel("价格")
    plt.ylabel("比例")
    frequencies = ax.get_yticks() / total_count
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(['{:.0%}'.format(x) for x in frequencies])
    plt.tight_layout()
    image_base64 = toBase64()
    return image_base64


def map_price_range(min_price=None, max_price=None, size="small"):
    df_map = df[["longitude", "latitude", "price"]]
    if min_price is not None and max_price is not None:
        df_map = df_map[(df_map['price'] >= min_price) & (df_map['price'] <= max_price)]
    elif min_price is not None:
        df_map = df_map[(df_map['price'] >= min_price)]
        max_price = df_map["price"].max()
    elif max_price is not None:
        df_map = df_map[(df_map['price'] <= max_price)]
        min_price = df_map['price'].min()
    else:
        max_price = df_map["price"].max()
        min_price = df_map['price'].min()
    if size == "medium":
        plt.figure(figsize=(10, 10))
    elif size == "large":
        plt.figure(figsize=(20, 20))
    plt.scatter(df_map["longitude"], df_map["latitude"], s=1, c=df_map["price"], cmap='tab20', marker='x')
    plt.xlim(121, 122)
    plt.ylim(30.7, 31.7)
    plt.xlabel("经度")
    plt.ylabel("纬度")
    plt.gca().set_aspect('equal')
    foot = int((max_price - min_price) / 20)
    if foot > 1000:
        foot = foot - foot % 1000
    plt.colorbar(label="price", ticks=range(0, max_price, foot)).set_label("价格")
    background_img = mpimg.imread("../static/map.png")
    plt.imshow(background_img, extent=(121.0, 122.0, 30.7, 31.7), aspect='auto', alpha=0.5)
    plt.tight_layout()
    image_base64 = toBase64()
    return image_base64


def KMeans_price(n_clusters, normalization_func, size="small"):
    kmeans = KMeans(n_clusters=n_clusters)
    df_price = df[['longitude', 'latitude', 'price']].copy()
    if normalization_func == 'n':
        df_price['price'] = normalization(df['price'].to_numpy())[0]
    elif normalization_func == 's':
        df_price['price'] = standardization(df['price'].to_numpy())[0]
    else:
        pass
    kmeans.fit(df_price)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    if size == "medium":
        plt.figure(figsize=(10, 10))
    elif size == "large":
        plt.figure(figsize=(20, 20))
    plt.scatter(df_price['longitude'], df_price['latitude'], s=1, c=labels)
    # plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='r')
    plt.xlim(121, 122)
    plt.ylim(30.7, 31.7)
    plt.xlabel("经度")
    plt.ylabel("纬度")
    background_img = mpimg.imread("../static/map.png")
    plt.imshow(background_img, extent=(121.0, 122.0, 30.7, 31.7), aspect='auto', alpha=0.5)
    plt.title('K-means Clustering')
    plt.tight_layout()
    image_base64 = toBase64()
    return image_base64


def DBSCAN_price(epsilon, min_samples, normalization_func, size="small"):
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    df_price = df[['longitude', 'latitude', 'price']].copy()
    if normalization_func == 'n':
        df_price['price'] = normalization(df['price'].to_numpy())[0]
    elif normalization_func == 's':
        df_price['price'] = standardization(df['price'].to_numpy())[0]
    else:
        pass
    dbscan.fit(df_price)
    labels = dbscan.labels_
    if size == "medium":
        plt.figure(figsize=(10, 10))
    elif size == "large":
        plt.figure(figsize=(20, 20))
    plt.scatter(df_price['longitude'], df_price['latitude'], s=1, c=labels)
    plt.xlim(121, 122)
    plt.ylim(30.7, 31.7)
    plt.xlabel("经度")
    plt.ylabel("纬度")
    background_img = mpimg.imread("../static/map.png")
    plt.imshow(background_img, extent=(121.0, 122.0, 30.7, 31.7), aspect='auto', alpha=0.5)
    plt.title('DBSCAN Clustering')
    plt.tight_layout()
    image_base64 = toBase64()
    return image_base64


def Agglomerative_price(n_clusters, linkage, normalization_func, size="small"):
    if linkage in ['ward', 'complete', 'average', 'single']:
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    else:
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    df_price = df[['longitude', 'latitude', 'price']].copy()
    if normalization_func == 'n':
        df_price['price'] = normalization(df['price'].to_numpy())[0]
    elif normalization_func == 's':
        df_price['price'] = standardization(df['price'].to_numpy())[0]
    else:
        pass
    agglomerative.fit(df_price)
    labels = agglomerative.labels_
    if size == "medium":
        plt.figure(figsize=(10, 10))
    elif size == "large":
        plt.figure(figsize=(20, 20))
    plt.scatter(df_price['longitude'], df_price['latitude'], s=1, c=labels)
    plt.xlim(121, 122)
    plt.ylim(30.7, 31.7)
    plt.xlabel("经度")
    plt.ylabel("纬度")
    background_img = mpimg.imread("../static/map.png")
    plt.imshow(background_img, extent=(121.0, 122.0, 30.7, 31.7), aspect='auto', alpha=0.5)
    plt.title('Agglomerative Clustering')
    plt.tight_layout()
    image_base64 = toBase64()
    return image_base64


if __name__ == "__main__":
    # hist_price_district("宝山")
    map_price_range(8000)
