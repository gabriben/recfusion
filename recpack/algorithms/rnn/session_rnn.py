from __future__ import annotations
import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch import Tensor

from sklearn.utils.validation import check_is_fitted

from recpack.algorithms.base import Algorithm
from recpack.algorithms.util import (
    StoppingCriterion,
    EarlyStoppingException,
    get_batches,
    get_users,
)
from recpack.metrics.recall import recall_k
from recpack.data.data_matrix import DataM
from typing import Tuple, Optional, Iterator

from recpack.algorithms.rnn.loss import (
    BatchSampler,
    BPRLoss,
    BPRMaxLoss,
    TOP1Loss,
    TOP1MaxLoss,
)
from recpack.algorithms.rnn.model import SessionRNNTorch


logger = logging.getLogger("recpack")


class SessionRNN(Algorithm):
    """
    A recurrent neural network for session-based recommendations.

    This algorithm is described in the 2016 and 2018 papers by Hidasi et al.
    "Session-based Recommendations with Recurrent Neural Networks" and
    "Recurrent Neural Networks with Top-k Gains for Session-based Recommendations"

    :param batch_size: Number of examples in a mini-batch
    :param sample_size: Number of negative samples used for loss calculation,
        including samples from the same minibatch
    :param alpha: Sampling weight parameter, 0 is uniform, 1 is popularity-based
    :param embedding_size: Size of item embeddings, None for no embeddings
    :param dropout: Dropout applied to embeddings and hidden layers
    :param num_layers: Number of hidden layers in the RNN
    :param hidden_size: Number of neurons in the hidden layer(s)
    :param activation: Final layer activation function, one of "identity", "tanh",
        "softmax", "relu", "elu-<X>", "leaky-<X>"
    :param loss_fn: Loss function, one of "cross-entropy", "top1", "top1-max",
        "bpr", "bpr-max"
    :param optimizer: Gradient descent optimizer, one of "sgd", "adagrad"
    :param learning_rate: Gradient descent initial learning rate
    :param momentum: Gradient descent momentum
    :param clip_norm: Clip the gradient's l2 norm, None for no clipping
    :param seed: Seed for random number generator
    :param bptt: Number of backpropagation through time steps
    :param num_epochs: Max training runs through entire dataset, None for no limit
    """

    def __init__(
        self,
        num_layers: int = 1,
        hidden_size: int = 100,
        embedding_size: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "elu-1",
        loss_fn: str = "bpr-max",
        sample_size: int = 5000,
        alpha: float = 0.5,
        optimizer: str = "adagrad",
        batch_size: int = 250,
        learning_rate: float = 0.03,
        momentum: float = 0.0,
        clip_norm: Optional[float] = 1.0,
        seed: int = 2,
        bptt: int = 1,
        num_epochs: Optional[int] = 20,
        stopping_criterion=None,  # TODO
    ):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.activation = activation
        self.loss_fn = loss_fn
        self.sample_size = sample_size
        self.alpha = alpha
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.clip_norm = clip_norm
        self.seed = seed
        self.bptt = bptt
        self.num_epochs = num_epochs
        self.stopping_criterion = stopping_criterion
        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.stopping_criterion = StoppingCriterion(
            recall_k, minimize=False, stop_early=False
        )

    def _init_random_state(self) -> None:
        """
        Resets the random number generator state.
        """
        if self.seed:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

    def _init_model(self, train_data: DataM) -> None:
        """
        Initializes the neural network model.
        """
        num_items = train_data.shape[1]

        self.model_ = SessionRNNTorch(
            num_items,
            self.hidden_size,
            num_layers=self.num_layers,
            embedding_size=self.embedding_size,
            dropout=self.dropout,
            activation=self.activation,
        ).to(self.device)

    def _init_training(self, train_data: DataM) -> None:
        """
        Initializes the objects required for network optimization.
        """
        item_counts = torch.as_tensor(
            train_data.timestamps.reset_index()["iid"]
            .value_counts()
            .sort_index()
            .to_numpy()
        )
        item_weights = item_counts ** self.alpha

        sampler = BatchSampler(item_weights, device=self.device)
        self._criterion = {
            "cross-entropy": nn.CrossEntropyLoss(),
            "top1": TOP1Loss(sampler, self.sample_size),
            "top1-max": TOP1MaxLoss(sampler, self.sample_size),
            "bpr": BPRLoss(sampler, self.sample_size),
            "bpr-max": BPRMaxLoss(sampler, self.sample_size),
        }[self.loss_fn]

        self._optimizer = {
            "sgd": optim.SGD(
                self.model_.parameters(), lr=self.learning_rate, momentum=self.momentum
            ),
            "adagrad": optim.Adagrad(self.model_.parameters(), lr=self.learning_rate),
        }[self.optimizer]

    def save(self, file) -> None:
        """
        Saves the algorithm, all its learned parameters and optimization state.

        :param file: A file-like or string containing a filepath.
        """
        torch.save(self, file)

    @staticmethod
    def load(file) -> SessionRNN:
        """
        Loads an algorithm previously stored with save().

        :param file: A file-like or string with the filepath of the algorithm
        """
        return torch.load(file)

    def fit(self, X: DataM, validation_data: Tuple[DataM, DataM] = None) -> SessionRNN:
        """
        Fit the model on the X dataset, and evaluate model quality on validation_data.

        :param train_data: The training data matrix.
        :param validation_data: Validation data, as matrix to be used as input and matrix to be used as output.
        """
        self._init_random_state()
        self._init_model(X)
        self._init_training(X)

        # TODO: verify train set & batch size compatible
        try:
            for epoch in range(self.num_epochs):
                logger.info(f"Epoch {epoch}")
                self._train_epoch(X)
                # self._evaluate(validation_data)
        except EarlyStoppingException:
            pass

        # self.load(self.stopping_criterion.best_value)  # Load best model
        return self

    def predict(self, X: DataM) -> csr_matrix:
        """
        Predict recommendations for each user with at least a single event in their history.

        :param X: Data matrix, should have same shape as training matrix
        """
        check_is_fitted(self)

        X_pred = lil_matrix(X.shape)

        with torch.no_grad():
            actions, _, uids = dm_to_tensor(
                X, batch_size=1, device=self.device, shuffle=True, include_last=True
            )
            new_uid = uids != uids.roll(1, dims=0)

            self.model_.train(False)
            hidden = self.model_.init_hidden(1)
            for i, (action, uid, new) in tqdm(
                enumerate(zip(actions, uids, new_uid)), total=len(actions)
            ):
                hidden = hidden * ~new.view(1, -1, 1)
                output, hidden = self.model_(action.unsqueeze(0), hidden)
                last_action = (i == len(actions) - 1) or new_uid[i + 1, 0]
                if last_action:
                    X_pred[uid[0].item()] = output.cpu().numpy().reshape(-1)

        X_pred = X_pred.tocsr()

        self._check_prediction(X_pred, X.values)

        return X_pred

    def _train_epoch(self, X: DataM) -> None:
        """
        Train model for a single epoch.
        """
        actions, targets, uids = dm_to_tensor(
            X, batch_size=self.batch_size, device=self.device, shuffle=True
        )
        new_uid = uids != uids.roll(1, dims=0)

        loss, losses = 0.0, []
        self.model_.train(True)
        hidden = self.model_.init_hidden(self.batch_size)
        for i, (action, target, new) in tqdm(
            enumerate(zip(actions, targets, new_uid)), total=len(actions)
        ):
            hidden = hidden * ~new.view(1, -1, 1)  # Reset hidden state between users
            output, hidden = self.model_(action.unsqueeze(0), hidden)
            loss += self._criterion(output, target) / self.bptt
            if i % self.bptt == self.bptt - 1:
                self._optimizer.zero_grad()
                loss.backward()
                if self.clip_norm:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.clip_norm)
                self._optimizer.step()
                losses.append(loss.item())
                loss = 0.0
                hidden = hidden.detach()  # Prevent backprop past bptt steps

        logger.info("training loss = {}".format(np.mean(losses)))

    def _evaluate(self, validation_data: Tuple[DataM, DataM]) -> None:
        """
        Evaluate the current model on the validation data.

        If performance improved over previous epoch, store the model and update
        best value. If performance stagnates, stop training.

        :param validation_data: validation data interaction matrix.
        """
        self.model_.eval()
        with torch.no_grad():
            X_val_pred = self.predict(validation_data[0])
            X_val_pred[
                validation_data[0].values.nonzero()
            ] = -np.inf  # TODO: should this belong in predict()?
            better = self.stopping_criterion.update(
                validation_data[1], X_val_pred, k=50
            )
            if better:
                self.save(self.stopping_criterion.best_value)
        # TODO: Log performance on val. Possible to read from stopping criterion

    def session_based_evaluate(self, X: DataM, K: int = 20):
        """
        Calculates Recall@K and MRR@K using the evaluation procedure from the 2016 paper.

        Each session is fed to the rnn action by action. If the true next item is in the
        top K predicted items it is considered a 'hit'. Recall is the fraction of hits
        over all actions in the dataset. MRR is the reciprocal rank of the true next item,
        averaged over all actions in the dataset.

        :param X: Session data to evaluate on
        :param K: Calculate metrics on top K recommendations
        """
        check_is_fitted(self)

        def recall(output, targets, K=K):
            _, indices = output.topk(K, dim=1)
            hits = (indices == targets.reshape(-1, 1)).sum().item()
            return hits / targets.nelement()

        def mrr(output, targets, K=K):
            _, indices = output.topk(K, dim=1)
            hits = (indices == targets.reshape(-1, 1)).nonzero()
            ranks = hits[:, -1] + 1
            reciprocal = torch.reciprocal(ranks.float())
            return (torch.sum(reciprocal) / targets.nelement()).item()

        with torch.no_grad():
            mean_recall, mean_mrr = self._session_evaluate(X, [recall, mrr])
            print("Recall: {:.5f}\nMRR: {:.5f}".format(mean_recall, mean_mrr))
            return mean_recall, mean_mrr

    def _session_evaluate(self, X: DataM, metrics: List):
        actions, targets, uids = dm_to_tensor(
            X, batch_size=1, device=self.device, shuffle=False
        )
        new_uid = uids != uids.roll(1, dims=0)

        self.model_.train(False)
        metric_scores = [[] for _ in metrics]
        hidden = self.model_.init_hidden(1)
        for action, target, new in tqdm(
            zip(actions, targets, new_uid), total=len(actions)
        ):
            hidden = hidden * ~new.view(1, -1, 1)  # Reset hidden state between users
            output, hidden = self.model_(action.unsqueeze(0), hidden)
            for m_idx, m in enumerate(metrics):
                score = m(output, target)
                metric_scores[m_idx].append(score)
        return [np.mean(scores) for scores in metric_scores]


def dm_to_tensor(
    dm: DataM,
    batch_size: int,
    device: str = "cpu",
    shuffle: bool = False,
    include_last: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Converts a data matrix with timestamp information to tensor format.

    As an example, with following interactions (sorted by time):
        uid  0    1      2      3
        iid  0 1  2 3 4  5 6 7  8 9
    The returned tensors for a batch size of two would be
        [[0, 5],      [[1, 6],        [[0, 2],
         [2, 6],       [3, 7],         [1, 2],
         [3, 8]]   ,   [4, 9]]   and   [1, 3]]
    Containing input item ids, next item ids, and user/session ids. Grouped by
    users, ordered by time. Users at boundaries may be split across columns.
    """
    # Convert the item and user ids to 1D tensors
    df = dm.timestamps.reset_index()
    if shuffle:
        df = shuffle_and_sort(df)
    else:
        df = df.sort_values(by=["uid", "ts"], ascending=True)
    iids = torch.tensor(df["iid"].to_numpy(), device=device)
    uids = torch.tensor(df["uid"].to_numpy(), device=device)
    # Drop the last action of each user if include_last is false
    if include_last:
        actions = iids
        targets = iids.roll(-1, dims=0)
    else:
        true = torch.tensor([True], device=device)
        is_first = torch.cat((true, uids[1:] != uids[:-1]))
        is_last = torch.cat((uids[:-1] != uids[1:], true))
        actions = iids[~is_last]
        targets = iids[~is_first]
        uids = uids[~is_last]
    # Create user-parallel mini batches
    return (
        batchify(actions, batch_size),
        batchify(targets, batch_size),
        batchify(uids, batch_size),
    )


def shuffle_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shuffle sessions but keep all interactions for a session together, sorted
    by timestamp.
    """
    # Generate a unique random number for each session/user
    uuid = df["uid"].unique()
    rand = pd.Series(data=np.random.permutation(len(uuid)), index=uuid, name="rand")
    df = df.join(rand, on="uid")
    # Shuffle sessions by sorting on their random number
    df = df.sort_values(by=["rand", "ts"], ascending=True, ignore_index=True)
    del df["rand"]
    return df


def batchify(data: Tensor, batch_size: int) -> Tensor:
    """
    Splits a sequence into contiguous batches, indexed along dim 0.
    """
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)
    data = data.view(batch_size, -1).t()
    data = data.contiguous()
    return data
