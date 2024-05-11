import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# 假设我们有以下数据集（每行是一个数据点，每列是一个特征）
data = np.array([
    [2.5, 2.4, 1.7],
    [0.5, 0.7, 1.7],
    [2.2, 2.9, 1.7],
    [1.9, 2.2, 1.7],
    [3.1, 3.0, 1.7],
    [2.3, 2.7, 1.7],
    [2.2, 1.6, 1.7],
    [1.5, 1.1, 1.7],
    [1.5, 1.6, 1.7],
    [1.1, 0.9, 1.7]
])

# 创建PCA对象，n_components设置为主成分的数量
pca = PCA(n_components=1)

# 对数据进行降维
reduced_data = pca.fit_transform(data)

# 打印降维后的数据
print("降维后的数据:")
print(reduced_data)

# 使用余弦相似度计算降维后数据点之间的相似度
similarity_matrix = cosine_similarity(reduced_data)

# 打印相似度矩阵
print("数据点之间的相似度矩阵:")
print(similarity_matrix)