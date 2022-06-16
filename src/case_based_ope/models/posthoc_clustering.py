import torch
import numpy as np
from case_based_ope.utils.postprocessing import model_compute
from scipy.cluster.vq import kmeans
from case_based_ope.utils.misc import compute_squared_distances
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

class PostHocClustering(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, n_clusters, lr_C=1):
        self.base_model = base_model
        self.n_clusters = n_clusters
        self.lr_C = lr_C

    def fit(self, X, y):
        encodings = model_compute(self.base_model, X)[-1]
        centroids = kmeans(encodings, self.n_clusters)[0]

        squared_distances = compute_squared_distances(torch.from_numpy(encodings), torch.from_numpy(centroids)).numpy()
        self.prototypes = encodings[np.argmin(squared_distances, axis=0)]

        squared_distances = compute_squared_distances(torch.from_numpy(encodings), torch.from_numpy(self.prototypes)).numpy()
        similarities = np.exp(np.negative(squared_distances))

        self.lr = LogisticRegression(max_iter=1000, C=self.lr_C)
        self.lr.fit(similarities, y)
        
        # In order to calibrate an estimator, it must have the attribute "classes_"
        self.classes_ = np.unique(y)
        
        return self

    def predict_proba(self, X):
        encodings = model_compute(self.base_model, X)[-1]
        squared_distances = compute_squared_distances(torch.from_numpy(encodings), torch.from_numpy(self.prototypes)).numpy()
        similarities = np.exp(np.negative(squared_distances))
        return self.lr.predict_proba(similarities)

    def predict(self, X):
        return self.predict_proba(X).argmax(-1)
