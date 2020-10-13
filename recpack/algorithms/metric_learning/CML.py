import logging
from typing import Tuple, List

import numpy as np
from scipy.sparse import csr_matrix

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import recall_score

from recpack.algorithms.base import Algorithm
from recpack.algorithms.util import StoppingCriterion, EarlyStoppingException

# TODO I decided to remove "use_rank_weight" because they always use it...

logger = logging.getLogger("recpack")


class CML(Algorithm):
    """
    Pytorch Implementation of
    [1] Cheng-Kang Hsieh et al., Collaborative Metric Learning. WWW2017
    http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf

    Version without features, referred to as CML in the paper.
    """

    def __init__(
        self,
        num_components,
        margin,
        learning_rate,
        clip_norm,
        use_cov_loss,
        num_epochs,
        seed,
        batch_size=50000,
        U=20,
    ):
        # TODO Figure out clip_norm?
        self.num_components = num_components
        self.margin = margin
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.use_cov_loss = use_cov_loss
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.U = U
        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

        # TODO Make this configurable
        self.stopping_criterion = StoppingCriterion(
            recall_score, minimize=False, stop_early=False
        )

    def _init_model(self, num_users, num_items):
        self.model_ = CMLTorch(
            num_users, num_items, num_components=self.num_components
        ).to(self.device)

        self.optimizer = optim.Adagrad(self.model_.parameters(), lr=self.learning_rate)

    def load(self, validation_loss):
        with open(f"{self.name}_loss_{validation_loss}.trch", "rb") as f:
            self.model_ = torch.load(f)

    def save(self, validation_loss):
        with open(f"{self.name}_loss_{validation_loss}.trch", "wb") as f:
            torch.save(self.model_, f)

    def fit(self, X: csr_matrix, validation_data: Tuple[csr_matrix, csr_matrix]):
        """Fit the model on the X dataset, and evaluate model quality on validation_data.

        :param X: The training data matrix.
        :type X: csr_matrix
        :param validation_data: The validation data matrix, should have same dimensions as X
        :type validation_data: csr_matrix
        """

        self._init_model(X.shape[0], X.shape[1])
        try:
            for epoch in range(self.num_epochs):
                self._train_epoch(X)
                self._evaluate(validation_data)
        except EarlyStoppingException:
            pass

        # Load the best of the models during training.
        self.load(self.stopping_criterion.best_value)
        return

    def predict(self, X: csr_matrix):
        """Predict recommendations for each user with at least a single event in their history.

        :param X: interaction matrix, should have same size as model.
        :type X: csr_matrix
        :raises an: [description]
        :return: csr matrix of same shape, with recommendations.
        :rtype: [type]
        """
        check_is_fitted(self)
        # TODO We can make it so that we can recommend for unknown users by giving them an embedding equal to the sum of all items viewed previously.
        # TODO Or raise an error
        assert X.shape == (self.model_.num_users, self.model_.num_items)
        users = list(set(X.nonzero()[0]))

        # TODO Implement

        # U = torch.LongTensor(users).to(self.device)
        # I = torch.arange(X.shape[1]).to(self.device)

        # TODO Make this more efficient
        # result = np.zeros(X.shape)
        # result[users] = self.model_.forward(U, I).detach().cpu().numpy()
        # return csr_matrix(result)

    def _train_epoch(self, train_data: csr_matrix):
        """train a single epoch. Uses sampler to generate samples,
        and loop through them in batches of self.batch_size.
        After each batch, update the parameters according to gradients.

        :param train_data: interaction matrix.
        :type train_data: csr_matrix
        """
        train_loss = 0.0
        self.model_.train()

        for users, positives_batch, negatives_batch in tqdm(
            warp_sample_pairs(train_data, U=self.U, batch_size=self.batch_size,)
        ):
            users = users.to(self.device)
            positives_batch = positives_batch.to(self.device)
            negatives_batch = negatives_batch.to(self.device)

            self.optimizer.zero_grad()

            dist_pos_interaction = self.model_.forward(users, positives_batch)
            dist_neg_interaction_flat = self.model_.forward(
                users.repeat_interleave(self.U),
                negatives_batch.reshape(self.batch_size * self.U, 1),
            )
            dist_neg_interaction = dist_neg_interaction_flat.reshape(self.batch_size, -1)
            loss = self._compute_loss(dist_pos_interaction, dist_neg_interaction)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        logger.info(f"training loss = {train_loss}")

    def _evaluate(self, validation_data: csr_matrix):
        """Perform evaluation step, samples get drawn
        from the validation data, and compute loss.

        If loss improved over previous epoch, store the model, and update best value.

        :param validation_data: validation data interaction matrix.
        :type validation_data: csr_matrix
        """
        # TODO Implement
        val_loss = 0.0
        self.model_.eval()
        with torch.no_grad():
            #     # Bootstrap 20% of the number of training samples from validation data.
            #     for d in warp_sample_pairs(
            #         validation_data,
            #         batch_size=self.batch_size,
            #         sample_size=validation_data.nnz,
            #     ):
            #         users = d[:, 0].to(self.device)
            #         target_items = d[:, 1].to(self.device)
            #         negative_items = d[:, 2].to(self.device)

            #         # TODO Maybe rename?
            #         positive_sim = self.model_.forward(users, target_items)
            #         negative_sim = self.model_.forward(users, negative_items)
            #         loss = self._compute_loss(positive_sim, negative_sim)
            #         val_loss += loss.item()
            logger.info(f"validation loss = {val_loss}")
            better = self.stopping_criterion.update(val_loss)

            if better:
                self.save()

    def _compute_loss(self, dist_pos_interaction, dist_neg_interaction):
        # TODO Implement loss function
        pass

def covariance_loss():
    # Their implementation really confuses me
    # X = tf.concat((self.item_embeddings, self.user_embeddings), 0)
    # n_rows = tf.cast(tf.shape(X)[0], tf.float32)
    # X = X - (tf.reduce_mean(X, axis=0))
    # cov = tf.matmul(X, X, transpose_a=True) / n_rows

    # return tf.reduce_sum(tf.matrix_set_diag(cov, tf.zeros(self.embed_dim, tf.float32))) * self.cov_loss_weight
    return 0


def warp_loss(dist_pos_interaction, dist_neg_interaction, margin):
    min_dist_neg_interaction, indices = dist_neg_interaction.min(dim=-1)

    pairwise_loss = margin + dist_pos_interaction - min_dist_neg_interaction

    # Implement the rank thingy


class CMLTorch(nn.Module):
    """
    # TODO Add description

    :param num_users: the amount of users
    :type num_users: int
    :param num_items: the amount of items
    :type num_items: int
    :param num_components: The size of the embedding per user and item, defaults to 100
    :type num_components: int, optional
    """

    def __init__(self, num_users, num_items, num_components=100):
        super().__init__()

        self.num_components = num_components
        self.num_users = num_users
        self.num_items = num_items

        self.W = nn.Embedding(num_users, num_components)  # User embedding
        self.H = nn.Embedding(num_items, num_components)  # Item embedding

        std = 1 / num_components ** 0.5
        # Initialise embeddings to a random start
        nn.init.normal_(self.W.weight, std=std)
        nn.init.normal_(self.H.weight, std=std)

        self.pdist = nn.PairwiseDistance(p=2)

    def forward(self, U: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        """
        Compute dot-product of user embedding (w_u) and item embedding (h_i)
        for every user and item pair in U and I.

        :param U: [description]
        :type U: [type]
        :param I: [description]
        :type I: [type]
        """
        w_U = self.W(U)
        h_I = self.H(I)

        # U and I are unrolled -> [u=0, i=1, i=2, i=3] -> [0, 1], [0, 2], [0,3]

        return self.pdist(w_U, h_I)


# TODO Integrate sampling methods somewhere more logical
def warp_sample_pairs(X: csr_matrix, U=10, batch_size=100):
    """
    Sample U negatives for every user-item-pair (positive).

    :param X: Interaction matrix
    :type X: csr_matrix
    :param U: Number of negative samples for each positive, defaults to 10
    :type U: int, optional
    :param batch_size: The number of samples returned per batch, defaults to 100
    :type batch_size: int, optional
    :yield: Iterator of torch.LongTensor of shape (batch_size, U+2). [User, Item, Negative Sample1, Negative Sample2, ...] 
    :rtype: Iterator[torch.LongTensor]
    """
    # Need positive and negative pair. Requires the existence of a positive for this item.
    positives = np.array(X.nonzero()).T  # As a (num_interactions, 2) numpy array
    num_positives = positives.shape[0]
    np.random.shuffle(positives)

    for start in range(0, num_positives, batch_size):
        batch = positives[start : start + batch_size]
        users = batch[:, 0]
        positives_batch = batch[:, 1]

        # Important only for final batch, if smaller than batch_size
        true_batch_size = min(batch_size, num_positives - start)

        negatives_batch = np.random.randint(0, X.shape[1], size=(true_batch_size, U))
        while True:
            # Fix the negatives that are equal to the positives, if there are any
            mask = np.apply_along_axis(
                lambda col: col == positives_batch, 0, negatives_batch
            )
            num_incorrect = np.sum(mask)

            if num_incorrect > 0:
                new_negatives = np.random.randint(0, X.shape[1], size=(num_incorrect,))
                negatives_batch[mask] = new_negatives
            else:
                # Exit the while loop
                break

        # sample_pairs_batch = np.empty((positives_batch.shape[0], U + 2))
        # sample_pairs_batch[:, :2] = positives_batch
        # sample_pairs_batch[:, 2:] = negatives_batch
        yield users, positives_batch, negatives_batch


class CMLWithFeatures(Algorithm):
    """
    Pytorch Implementation of
    [1] Cheng-Kang Hsieh et al., Collaborative Metric Learning. WWW2017
    http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf

    Version with features, referred to as CML+F in the paper.
    """

    def __init__(
        self,
        embedding_dim,
        margin,
        learning_rate,
        clip_norm,
        use_cov_loss,
        hidden_layer_dim,
        feature_l2_reg,
        feature_proj_scaling_factor,
    ):
        pass

    def fit(self, X):
        pass

    def predict(self, X):
        pass
