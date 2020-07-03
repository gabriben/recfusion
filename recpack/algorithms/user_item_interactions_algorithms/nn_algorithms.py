import numpy as np
import scipy
from scipy.sparse import diags
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.validation import check_is_fitted

from recpack.algorithms.user_item_interactions_algorithms import (
    SimilarityMatrixAlgorithm,
)


class ItemKNN(SimilarityMatrixAlgorithm):

    def __init__(self, K=200):
        """Construct an ItemKNN model. Before use make sure to fit the model.
        The K parameter defines the how much best neighbours are kept for each item."""
        super().__init__()
        self.K = K

    def fit(self, X, y=None):
        """Fit a cosine similarity matrix from item to item"""
        # Do the cosine similarity computation here, this way we can set the diagonal to zero
        # to avoid self recommendation
        # X.T otherwise we are doing a user KNN
        self.item_cosine_similarities_ = cosine_similarity(X.T, dense_output=False)

        self.item_cosine_similarities_.setdiag(0)
        # Set diagonal to 0, because we don't want to support self similarity

        # resolve top K per item
        # Get indices of top K items per item
        indices = [
            (i, j)
            for i, best_items_row in enumerate(np.argpartition(self.item_cosine_similarities_.toarray(), -self.K))
            for j in best_items_row[-self.K:]
        ]
        # Create a mask matrix which will be pointwise multiplied with the similarity matrix.
        mask = scipy.sparse.csr_matrix(([1 for i in range(len(indices))], (list(zip(*indices)))))
        self.item_cosine_similarities_ = self.item_cosine_similarities_.multiply(mask)
        return self

    def get_sim_matrix(self):
        return self.item_cosine_similarities_


class NotItemKNN(SimilarityMatrixAlgorithm):
    """
    TODO: Figure out what this code is actually implementing. It is not cosine similarity
    It does seem to work fine though.
    """

    def __init__(self, K=200):
        """Construct an ItemKNN model. Before use make sure to fit the model.
        The K parameter defines the how much best neighbours are kept for each item."""
        super().__init__()
        self.K = K

    def fit(self, X):
        """Fit a cosine similarity matrix from item to item"""
        co_mat = X.T @ X
        # Do the cosine similarity computation here, this way we can set the diagonal to zero
        # to avoid self recommendation
        A = diags(1 / co_mat.diagonal())

        # This has all item-cosine similarities. Now we should probably set N-K to zero
        self.item_cosine_similarities_ = A @ co_mat

        # Set diagonal to 0, because we don't support self similarity
        self.item_cosine_similarities_.setdiag(0)

        # resolve top K per item
        # Get indices of top K items per item
        indices = [
            (i, j)
            for i, best_items_row in enumerate(np.argpartition(self.item_cosine_similarities_.toarray(), -self.K))
            for j in best_items_row[-self.K:]
        ]
        # Create a mask matrix which will be pointwise multiplied with the similarity matrix.
        mask = scipy.sparse.csr_matrix(([1 for i in range(len(indices))], (list(zip(*indices)))))
        self.item_cosine_similarities_ = self.item_cosine_similarities_.multiply(mask)
        return self

    def get_sim_matrix(self):
        return self.item_cosine_similarities_

