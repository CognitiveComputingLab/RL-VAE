import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import matplotlib.pyplot as plt



def get_knn_score(embeddings, labels):
    # get KNN score
    n_neighbors = 6
    knn = KNeighborsClassifier(n_neighbors=n_neighbors + 1)
    knn.fit(embeddings, labels)
    distances, indices = knn.kneighbors(embeddings)
    # exclude the point itself and predict the label

    # predicted_labels = np.array([mode(labels[indices[i][1:]])[0][0] for i in range(len(labels))])
    predicted_labels = [mode(labels[indices[i][1:]])[0] for i in range(len(labels))]

    # calculate accuracy
    accuracy = np.mean(predicted_labels == labels)
    return accuracy


def get_scores(file_path):
    # load all raw data
    data = np.load(file_path)
    high_dim_data = data["high_dim_data"]

    # loop through each epoch
    for epoch in range(data["embeddings"].shape[0]):
        # get epoch data
        embeddings = data["embeddings"][epoch]
        colors = data["colors"][epoch]
        labels = data["labels"][epoch]

        print(f"------------ {epoch} ------------")
        print(embeddings.shape)
        print(labels.shape)

        knn_accuracy = get_knn_score(embeddings, labels)
        print("KNN accuracy: ", knn_accuracy)

        sil_score = silhouette_score(embeddings, labels)
        print("silhouette score: ", sil_score)

        trust_score = trustworthiness(high_dim_data, embeddings, n_neighbors=5)
        print("trustworthiness score: ", trust_score)

        original_distances = pdist(high_dim_data)
        embedded_distances = pdist(embeddings)
        correlation, _ = pearsonr(original_distances, embedded_distances)
        print("shepard diagram correlation: ", correlation)


if __name__ == "__main__":
    get_scores("images/raw-data.npz")


