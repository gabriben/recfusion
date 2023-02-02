import logging
from typing import List, Tuple
from scipy.sparse.lil import lil_matrix

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from scipy.sparse import csr_matrix
import numpy as np

from recpack.algorithms.base import TorchMLAlgorithm
from recpack.algorithms.loss_functions import vae_loss
from recpack.algorithms.util import naive_sparse2tensor
from recpack.scenarios.splitters import yield_batches, yield_batches_same_size

from recpack.algorithms import ddpm
from recpack.algorithms.util import get_batches, get_users, sample_rows
from recpack.matrix import InteractionMatrix, to_csr_matrix, Matrix


logger = logging.getLogger("recpack")


class CODIGEM(TorchMLAlgorithm):
    """MultVAE Algorithm as first discussed in
    'Variational Autoencoders for Collaborative Filtering',
    D. Liang et al. @ KDD2018.

    An Auto Encoder neural network's goal is to reconstruct the original matrix after
    being passed through a bottleneck layer and several hidden layers.
    This method assumes a Multinomial likelihood for the data distribution.
    This rewards the model for putting probability mass on the non-zero entries in x_u.
    But the model has a limited budget of probability mass since π(z_u) must sum to 1;
    the items must compete for a limited budget.

    Default values for parameters were taken from the paper.

    :param batch_size: Batch size for SGD,
                        defaults to 500
    :type batch_size: int, optional
    :param max_epochs: Maximum number of epochs (iterations),
                        defaults to 200
    :type max_epochs: int, optional
    :param learning_rate: Learning rate, defaults to 1e-4
    :type learning_rate: [type], optional
    :param seed: Random seed for Torch, provided for reproducibility,
            defaults to None.
    :type seed: int, optional
    :param dim_bottleneck_layer: Size of the latent representation,
                                    defaults to 200
    :type dim_bottleneck_layer: int, optional
    :param dim_hidden_layer: Dimension of the hidden layer, defaults to 600
    :type dim_hidden_layer: int, optional
    :param max_beta: Regularization parameter, annealed over ``anneal_steps``
                    until it reaches max_beta, defaults to 0.2
    :type max_beta: float, optional
    :param anneal_steps: Number of steps to anneal beta to ``max_beta``,
                            defaults to 200000
    :type anneal_steps: int, optional
    :param dropout: Dropout rate to apply at the inputs, defaults to 0.5
    :type dropout: float, optional
    :param stopping_criterion: Used to identify the best model computed thus far.
        The string indicates the name of the stopping criterion.
        Which criterions are available can be found at StoppingCriterion.FUNCTIONS
        Defaults to ``'ndcg'``
    :type stopping_criterion: str, optional
    :param stop_early: If True, early stopping is enabled,
        and after ``max_iter_no_change`` iterations where improvement of loss function
        is below ``min_improvement`` the optimisation is stopped,
        even if max_epochs is not reached.
        Defaults to False
    :type stop_early: bool, optional
    :param max_iter_no_change: If early stopping is enabled,
        stop after this amount of iterations without change.
        Defaults to 5
    :type max_iter_no_change: int, optional
    :param min_improvement: If early stopping is enabled, no change is detected,
        if the improvement is below this value.
        Defaults to 0.01
    :param save_best_to_file: If True, the best model is saved to disk after fit.
    :type save_best_to_file: bool, optional
    :param keep_last: Retain last model, rather than best
        (according to stopping criterion value on validation data), defaults to False
    :type keep_last: bool, optional
    :param predict_topK: The topK recommendations to keep per row in the matrix.
        Use when the user x item output matrix would become too large for RAM.
        Defaults to None, which results in no filtering.
    :type predict_topK: int, optional
    :param validation_sample_size: Amount of users that will be sampled to calculate
        validation loss and stopping criterion value.
        This reduces computation time during validation, such that training times are strongly reduced.
        If None, all nonzero users are used. Defaults to None.
    :type validation_sample_size: int, optional
    """

    def __init__(
        self,
        stopping_criterion: str = "ndcg",
        stop_early: bool = True,
        max_iter_no_change: int = 5,
        min_improvement: int = 0.01,
        save_best_to_file=False,
        keep_last: bool = False,
        predict_topK: int = None,
        validation_sample_size: int = None,
        batch_size: int = 200,
        max_epochs: int = 200,
        learning_rate: float = 1e-4,
        seed: int = None,

        # learner: "adamax"

        anneal_cap: int = 1,
        total_anneal_steps: int = 0,
        T: int = 3,
        M: int = 200,
        b_start: float = 0.0001,
        b_end: float = 0.1,
        schedule_type: str = "quadratic",
        reduction: str = "avg",

        # xavier_initialization: bool = False,
        x_to_negpos: bool = False,
        # decode_from_noisiest: bool = False,
        p_dnns_depth: int = 4,
        # decoder_net_depth: int = 4
    ):

        super().__init__(
            batch_size,
            max_epochs,
            learning_rate,
            stopping_criterion,
            stop_early=stop_early,
            max_iter_no_change=max_iter_no_change,
            min_improvement=min_improvement,
            seed=seed,
            save_best_to_file=save_best_to_file,
            keep_last=keep_last,
            predict_topK=predict_topK,
            validation_sample_size=validation_sample_size,
        )

        self.b_start = b_start
        self.b_end = b_end
        self.schedule_type = schedule_type
        betas = self.get_beta_schedule(b_start, b_end, T, schedule_type)
        self.betas = torch.FloatTensor(betas).to(self.device)

        self.steps = 0

        self.anneal_cap = anneal_cap
        self.total_anneal_steps = total_anneal_steps
        self.T = T
        self.M = M
        self.schedule_type = schedule_type,
        self.reduction = reduction,
        self.x_to_negpos = x_to_negpos
        self.update = 0

        self.p_dnns_depth = p_dnns_depth
        
        # self.dropout = dropout

        # self.optimizer = None
        # self.loss_function = vae_loss

    # https://github.com/InFoCusp/diffusion_models/blob/main/Diffusion_models.ipynb
    def get_beta_schedule(self, b_start, b_end, T, schedule_type):
        if schedule_type == 'quadratic':
            betas = np.linspace(b_start ** 0.5, b_end ** 0.5, T, dtype=np.float32) ** 2
        elif schedule_type == 'linear':
            betas = np.linspace(b_start, b_end, T, dtype=np.float32)
        return betas

    def _init_model(self, X: csr_matrix):
        """
        Initialize Torch model and optimizer.

        :param dim_input_layer: Dimension of the input layer
                                (corresponds to number of items)
        :type dim_input_layer: int
        """

        D = X.shape[1] # number of items

        # self.model_ = OriginalUnet(dim = 1, channels = 1, resnet_block_groups=1, dim_mults=(1, 2)).to(self.device)

        # self.mlp = MLP(self.p_dnns_depth, D, self.M).to(self.device)
        self.model_ = MLPPerStep(self.p_dnns_depth, self.T, D, self.M).to(self.device)

        pdb.set_trace()

        # params = list(self.mlp.parameters()) + list(self.mlp_step.parameters())
        self.optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

    def _train_epoch(self, train_data: csr_matrix):
        """
        Perform one training epoch.
        Data is processed in batches of self.batch_size users.

        :param train_data: Training data (UxI)
        :type train_data: [type]
        """
        losses = []
        
        users = list(set(train_data.nonzero()[0]))

        # if len(user) < 200:       
        #     user = torch.cat((user, self.prev_users), 0)[:200]

        # self.prev_users = user  

        np.random.shuffle(users)

        for batch_idx, user_batch in enumerate(yield_batches_same_size(users, self.batch_size)):
            X = naive_sparse2tensor(train_data[user_batch, :]).to(self.device)

            # Clear gradients
            self.optimizer.zero_grad()            

            if self.x_to_negpos:
                X = (X - 0.5) * 2

            # =====
            # forward difussion

            Z = [make_noise(X, 0, self.betas[0])]

            for i in range(1, self.T):
                Z.append(make_noise(Z[-1], i, self.betas[i]))  

            # =====
            # backward diffusion

            Z_mu_hat = []
            Z_var_hat = []

            for t in range(self.T):
                h = self.model_.m[t](Z[t])
                Z_mu_hat_i, Z_var_hat_i = torch.chunk(h, 2, dim=1)

                Z_mu_hat.append(Z_mu_hat_i)
                Z_var_hat.append(Z_var_hat_i)

            X_hat = self.model_.m[-1](Z[0])

            self.update += 1
            if self.total_anneal_steps > 0:
                anneal = min(self.anneal_cap, 1. * self.update / self.total_anneal_steps)
            else:
                anneal = self.anneal_cap
            
            loss = self._compute_loss(X, X_hat,
                                      Z, Z_mu_hat, Z_var_hat,
                                      anneal)
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()

            self.steps += 1

        return losses

    def _compute_loss(
        self,
        X: torch.Tensor,
        X_hat: torch.Tensor,
        Z: list,            
        Z_mu_hat: list,
        Z_var_hat: list,            
        anneal: int
    ) -> torch.Tensor:
        """Compute the prediction loss.

        More info on the loss function in the paper

        :param X: input data
        :type X: torch.Tensor
        :param X_pred: output data
        :type X_pred: torch.Tensor
        :param mu: the mean tensor
        :type mu: torch.Tensor
        :param logvar: the variance tensor
        :type logvar: torch.Tensor
        :return: the loss tensor
        :rtype: torch.Tensor
        """

        # =====ELBO
        # RE

        # Normal RE
        RE = log_standard_normal(X - X_hat[0]).sum(-1)

        # KL
        KL = (log_normal_diag(Z[-1], torch.sqrt(1. - self.betas[0]) * Z[-1],
                              torch.log(self.betas[0])) - log_standard_normal(Z[-1])).sum(-1)
        
        for i in range(len(Z)):
            KL_i = (log_normal_diag(Z[i], torch.sqrt(1. - self.betas[i]) * Z[i],
                                    torch.log(self.betas[i])) -
                    log_normal_diag(Z[i], Z_mu_hat[i], Z_var_hat[i])).sum(-1)

            KL = KL + KL_i

        # Final ELBO

        if self.reduction == 'sum':
            loss = -(RE - anneal * KL).sum()
        else:
            loss = -(RE - anneal * KL).mean()

        return loss

    def _batch_predict(self, X: csr_matrix, users: List[int]) -> csr_matrix:
        """Predict scores for matrix X, given the selected users in this batch

        :param X: Matrix of user item interactions,
            expected to only contain interactions for those users that are in `users`
        :type X: csr_matrix
        :param users: users selected for recommendation
        :type users: List[int]
        :return: Sparse matrix of scores per user item pair.
        :rtype: csr_matrix
        """

        active_users = X[users]
            
        in_tensor = naive_sparse2tensor(active_users).to(self.device)

        out_tensor = self.model_.m[-1](in_tensor)

        result = lil_matrix(X.shape)
        result[users] = out_tensor.detach().cpu().numpy()

        return result.tocsr()


    # def _predict(self, X: Matrix) -> csr_matrix:
    #     """Compute predictions per batch of users,
    #     to avoid going out of RAM on the GPU

    #     Will batch the nonzero users into batches of self.batch_size.

    #     :param X: The input user interaction matrix
    #     :type X: csr_matrix
    #     :return: The predicted affinity of users for items.
    #     :rtype: csr_matrix
    #     """
        
    #     results = lil_matrix(X.shape)
    #     self.model_.eval()
    #     with torch.no_grad():
    #         for users in get_batches(get_users(X), batch_size=self.batch_size):
    #             if isinstance(X, InteractionMatrix):
    #                 batch = X.users_in(users)
    #             else:
    #                 batch = lil_matrix(X.shape)
    #                 batch[users] = X[users]
    #                 batch = batch.tocsr()

    #             results[users] = self._get_top_k_recommendations(self._batch_predict(batch, users=users)[users])

    #     logger.debug(f"shape of response ({results.shape})")

    #     return results.tocsr()    

#### CODIGEM
    
PI = torch.from_numpy(np.asarray(np.pi))    

def make_noise(x, i, B):
    return torch.sqrt(1. - B) * x + torch.sqrt(B) * torch.randn_like(x)
    
def log_standard_normal(x, reduction=None, dim=None):
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * x**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * log_var - \
        0.5 * torch.exp(-log_var) * (x - mu)**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p    


#### MLP net


# class MLP(nn.Module):
#     def __init__(self, depth, D, M):
#         super().__init__()
#         self.m = nn.Sequential(
#             *[nn.Linear(D, M), nn.PReLU()] +
#             [nn.Linea\r(M, M), nn.PReLU()] * depth +
#             [nn.Linear(M, D), nn.Tanh()])
        
#     def forward(self, x):
#         return self.m(x)


class MLPPerStep(nn.Module):
    def __init__(self, depth, steps, D, M):
        super().__init__()
        self.m = nn.ModuleList([nn.Sequential(
            *[nn.Linear(D, M), nn.PReLU()] +
            [nn.Linear(M, M), nn.PReLU()] * depth +
            [nn.Linear(M, 2*D)]) for _ in range(steps)] +
                               [nn.Sequential(
            *[nn.Linear(D, M), nn.PReLU()] +
            [nn.Linear(M, M), nn.PReLU()] * depth +
            [nn.Linear(M, D), nn.Tanh()])])
        
    def forward(self, x):
        return self.m(x)
