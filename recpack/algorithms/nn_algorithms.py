from collections import Counter, defaultdict
import math
import numpy as np
import scipy
from scipy.sparse import diags
from sklearn.metrics.pairwise import cosine_similarity


from .algorithm_base import Algorithm


class ItemKNN(Algorithm):

    def __init__(self, K=200):
        """Construct an ItemKNN model. Before use make sure to fit the model.
        The K parameter defines the how much best neighbours are kept for each item."""
        self.K = K
        self.item_cosine_similarities = None

    def fit(self, X):
        """Fit a cosine similarity matrix from item to item"""
        # Do the cosine similarity computation here, this way we can set the diagonal to zero
        # to avoid self recommendation
        # X.T otherwise we are doing a user KNN
        self.item_cosine_similarities = cosine_similarity(X.T, dense_output=False)

        # Set diagonal to 0, because we don't want to support self similarity
        self.item_cosine_similarities.setdiag(0)

        # resolve top K per item
        # Get indices of top K items per item
        indices = [
            (i, j)
            for i, best_items_row in enumerate(np.argpartition(self.item_cosine_similarities.toarray(), -self.K))
            for j in best_items_row[-self.K:]
        ]
        # Create a mask matrix which will be pointwise multiplied with the similarity matrix.
        mask = scipy.sparse.csr_matrix(([1 for i in range(len(indices))], (list(zip(*indices)))))
        self.item_cosine_similarities = self.item_cosine_similarities.multiply(mask)

    def predict(self, X):
        # Use total sum of similarities
        # TODO: Use average?
        if self.item_cosine_similarities is None:
            raise Exception("Fit a model before trying to predict with it.")
        scores = X @ self.item_cosine_similarities
        return scores.toarray()

    @property
    def name(self):
        return f"item_knn_{self.K}"


class SharedAccount(ItemKNN):

    def __init__(self, K):
        super().__init__(K)

    def predict(self, X):
        raise NotImplementedError("Under construction, the gnomes are working on it.")
