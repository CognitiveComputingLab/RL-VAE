import numpy as np
from toy_data import embedding
from pitchscapes.keyfinding import KeyEstimator
from pitchscapes.plotting import key_scores_to_color


class DummyData:
    def __init__(self):
        self.data_path = "data/data_WTK_r200_p0.0_s0.03_c0_sparse_corpus.npy"
        self.all_data = np.load(self.data_path)
        # self.data = self.all_data[:100000, :]
        self.data = self.all_data

        self.k = KeyEstimator()
        self.scores = self.k.get_score(self.data)
        self.colors = key_scores_to_color(self.scores, circle_of_fifths=True)


sparse_corpus = DummyData()
print(sparse_corpus.data)
umap = embedding.UMAP(sparse_corpus)
umap.fit()
umap.plot()
