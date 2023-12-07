from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd

# 1. 数据准备
df = pd.read_csv("../static/Processed.csv", encoding="utf-8-sig")
sns.set_style("darkgrid")
sns.set_palette(sns.hls_palette(n_colors=14))
plt.rcParams['font.sans-serif'] = 'DengXian'

n_clusters = 100
normalization_func = "n"


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

# # 2. 使用聚类算法进行聚类分析
# kmeans = KMeans(n_clusters=3)
# kmeans.fit(X)
# cluster_labels = kmeans.labels_
#
# 3. 将簇标签作为预测模型的目标变量
X_train, X_test, y_train, y_test = train_test_split(df_price[['longitude', 'latitude']], labels, test_size=0.2,
                                                    random_state=42)

# 4. 使用其他的机器学习算法进行建模和预测
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#
# # 5. 模型评估
# accuracy = model.score(X_test, y_test)
# plt.show()
