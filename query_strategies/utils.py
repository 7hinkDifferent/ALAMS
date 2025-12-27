import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans

def rank_based_normalization(scores):
    """Rank based normalization to [0,1]
    Higher score means more important
    """
    ranks = scores.argsort().argsort() # ascending order
    normalized_scores = ranks / (len(scores) - 1)
    return normalized_scores

# compute gaussian scores based on kmeans clustering
# for each cluster, approximate the distribution of the embeddings with a gaussian distribution
# mean is cluster center, std^2 is \sum{|x_i - mu|^2} / N
# compute the score with exp{-|x-mu|^2 / (2*sigma^2)}

def kmeans_gaussian_scores(embeddings, n):
    # KMeans 聚类
    cluster_learner = KMeans(n_clusters=n)
    cluster_learner.fit(embeddings)
    
    cluster_idxs = cluster_learner.predict(embeddings)
    centers = cluster_learner.cluster_centers_
    
    # 计算每个样本到所属簇中心的距离
    # |x - mu|^2
    dis = np.sum((embeddings - centers[cluster_idxs])**2, axis=1)
    
    # 计算每个簇的方差 sigma^2 = \sum{|x_i - mu|^2} / N
    variances = np.zeros(n)
    for i in range(n):
        cluster_mask = cluster_idxs == i
        if cluster_mask.sum() > 0:
            cluster_dis = dis[cluster_mask]
            variances[i] = cluster_dis.sum() / cluster_mask.sum()
            # 避免方差为0的情况
            if variances[i] < 1e-10:
                variances[i] = 1e-10
    
    # 计算高斯得分: exp{-|x-mu|^2 / (2*sigma^2)}
    gaussian_scores = np.exp(-dis / (2 * variances[cluster_idxs]))
    return gaussian_scores, {
        "cluster_idxs": cluster_idxs.tolist(),
        "means": centers.tolist(),
        "variances": variances.tolist(),
    }


# gaussian mixture model scores based on kmeans clustering
# for each cluster, approximate the distribution of the embeddings with a gaussian distribution
# mean is cluster center, std^2 is \sum{|x_i - mu|^2} / N
# compute the prior probability of each cluster
# compute the score with \sum{prior * exp{-|x-mu|^2 / (2*sigma^2)}}
def kmeans_gmm_scores(embeddings, n):
    
    _, cluster_info = kmeans_gaussian_scores(embeddings, n)
    cluster_idxs = np.array(cluster_info["cluster_idxs"])
    means = np.array(cluster_info["means"])
    variances = np.array(cluster_info["variances"])

    # 计算每个簇的先验概率
    priors = np.zeros(n)
    for i in range(n):
        cluster_mask = cluster_idxs == i
        priors[i] = cluster_mask.sum() / len(embeddings)

    # 计算GMM得分: \sum{prior * exp{-|x-mu|^2 / (2*sigma^2)}}
    gmm_scores = np.zeros(len(embeddings))
    for i in range(n):
        dis = np.sum((embeddings - means[i])**2, axis=1)
        gmm_scores += priors[i] * np.exp(-dis / (2 * variances[i]))
    
    return gmm_scores, {
        "cluster_idxs": cluster_idxs.tolist(),
        "means": means.tolist(),
        "variances": variances.tolist(),
        "priors": priors.tolist(),
    }

