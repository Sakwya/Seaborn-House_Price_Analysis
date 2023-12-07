from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# 1. 数据准备
X = df[['特征1', '特征2', '特征3']]
y = df['簇标签']

# 2. 使用聚类算法进行聚类分析
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
cluster_labels = kmeans.labels_

# 3. 将簇标签作为预测模型的目标变量
X_train, X_test, y_train, y_test = train_test_split(X, cluster_labels, test_size=0.2, random_state=42)

# 4. 使用其他的机器学习算法进行建模和预测
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. 模型评估
accuracy = model.score(X_test, y_test)
