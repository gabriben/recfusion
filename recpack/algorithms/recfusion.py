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


class RecFusion(TorchMLAlgorithm):
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

        self.model_ = OriginalUnet(dim = 1, channels = 1, resnet_block_groups=1, dim_mults=(1, 2)).to(self.device)

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

            if self.x_to_negpos:
                X = (X - 0.5) * 2

            # =====
            # forward difussion

            Z = [make_noise(X, 0, self.betas[0])]

            for i in range(1, self.T):
                Z.append(make_noise(Z[-1], i, self.betas[i]))                

            # =====
            # backward diffusion

            Z_hat = []

            for t in range(self.T - 1):
                t = torch.tensor([t], dtype=torch.int32).to(self.device)
                mu_t = self.model_.forward(Z[t+1][None, None, :, :], t+1)   
                Z_hat.append(mu_t)


            self.update += 1
            if self.total_anneal_steps > 0:
                anneal = min(self.anneal_cap, 1. * self.update / self.total_anneal_steps)
            else:
                anneal = self.anneal_cap

            # Clear gradients
            self.optimizer.zero_grad()
            
            loss = self._compute_loss(X, Z_hat, anneal)
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()

            self.steps += 1

        return losses

    def _compute_loss(
        self,
        X: torch.Tensor,
        Z_hat: torch.Tensor,
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
        # RE = log_standard_normal(x - mu_x).sum(-1)
        RE = log_standard_normal(X - Z_hat[0]).sum(-1)

        # KL
        KL = (log_normal_diag(Z_hat[-1], torch.sqrt(1. - self.betas[0]) * Z_hat[-1],
                              torch.log(self.betas[0])) - log_standard_normal(Z_hat[-1])).sum(-1)

        for i in range(1, len(Z_hat) - 1):
            KL_i = (log_normal_diag(Z_hat[i], torch.sqrt(1. - self.betas[i]) * Z_hat[i],
                              torch.log(self.betas[i])) - log_standard_normal(Z_hat[i])).sum(-1)
        
        # for i in range(len(self.mus)):
        #     KL_i = (log_normal_diag(self.Z[i], torch.sqrt(1. - self.betas[i]) * self.Z[i], torch.log(
        #         self.betas[i])) - log_normal_diag(self.Z[i], self.mus[i], self.log_vars[i])).sum(-1)

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

        t = torch.tensor([1], dtype=torch.int32).to(self.device)
        
        out_tensor = self.model_(in_tensor[None, None, :, :], t)

        result = lil_matrix(X.shape)
        result[users] = out_tensor.detach().cpu().numpy()

        return result.tocsr()


    def _predict(self, X: Matrix) -> csr_matrix:
        """Compute predictions per batch of users,
        to avoid going out of RAM on the GPU

        Will batch the nonzero users into batches of self.batch_size.

        :param X: The input user interaction matrix
        :type X: csr_matrix
        :return: The predicted affinity of users for items.
        :rtype: csr_matrix
        """

        n_users = len(get_users(X))
        rep_users = self.batch_size - n_users % self.batch_size
        users_modulo_batch = get_users(X) + get_users(X)[:rep_users]
        
        results = lil_matrix(X.shape)
        self.model_.eval()
        
        with torch.no_grad():
            for users in get_batches(users_modulo_batch, batch_size=self.batch_size):
                if isinstance(X, InteractionMatrix):
                    batch = X.users_in(users)
                else:
                    batch = lil_matrix(X.shape)
                    batch[users] = X[users]
                    batch = batch.tocsr()
                
                results[users] = self._get_top_k_recommendations(self._batch_predict(batch, users=users)[users])

        logger.debug(f"shape of response ({results.shape})")

        return results.tocsr()    

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


class MLP(nn.Module):
    def __init__(self, depth, D, M):
        super().__init__()
        self.m = nn.Sequential(
            *[nn.Linear(D, M), nn.PReLU()] +
            [nn.Linear(M, M), nn.PReLU()] * depth +
            [nn.Linear(M, D), nn.Tanh()])
        
    def forward(self, x):
        return self.m(x)

# def MLP(depth, D, M):
#     return
#     nn.Sequential(
#         *[nn.Linear(D, M), nn.PReLU()] +
#         [nn.Linear(M, M), nn.PReLU()] * depth +
#         [nn.Linear(M, D), nn.Tanh()])


class MLPperStep(nn.Module):
    def __init__(self, depth, steps, D, M):
        super().__init__()
        self.m = nn.ModuleList([nn.Sequential(
            *[nn.Linear(D, M), nn.PReLU()] +
            [nn.Linear(M, M), nn.PReLU()] * depth +
            [nn.Linear(M, 2*D)]) for _ in range(steps-1)])
        
    def forward(self, x, t):
        return self.m(x, t)


#### DDPM Unet
    
import copy
import logging
import math
from collections import namedtuple
from functools import partial
from math import log as ln
from math import pi, sqrt
from multiprocessing import cpu_count
from pathlib import Path
from random import random

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from PIL import Image
from torch import einsum, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision import utils
from tqdm import tqdm
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

import pdb

# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# small helper modules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb"""

    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, dropout=0.0):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        x = self.dropout(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, dropout=0.0):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups, dropout=0.0)
        self.block2 = Block(dim_out, dim_out, groups=groups, dropout=dropout)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), LayerNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


# model


class OriginalUnet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        dropout=0.1,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups, dropout=dropout)

        # time embeddings

        time_dim = dim * 4
        # time_dim = dim * 1        

        sinu_pos_emb = SinusoidalPosEmb(time_dim)        
        fourier_dim = time_dim
        
        # sinu_pos_emb = SinusoidalPosEmb(dim)
        # fourier_dim = dim
      

        # pdb.set_trace()

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, Attention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, Attention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time):
        
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []
        
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            # h.append(x)

            # x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        # x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            # x = torch.cat((x, h.pop()), dim=1)
            # x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x) -> torch.Tensor:
    return 0.5 * (1.0 + torch.tanh(sqrt(2.0 / pi) * (x + 0.044715 * (x ** 3))))


def discretized_gaussian_log_likelihood(x, means, log_scales):
    # Assumes data is integers [0, 255] rescaled to [-1, 1]
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(
        torch.maximum(cdf_plus, torch.tensor(1e-12, device=cdf_plus.device))
    )
    log_one_minus_cdf_min = torch.log(
        torch.maximum(1.0 - cdf_min, torch.tensor(1e-12, device=cdf_min.device))
    )
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999,
            log_one_minus_cdf_min,
            torch.log(
                torch.maximum(cdf_delta, torch.tensor(1e-12, device=cdf_delta.device))
            ),
        ),
    )
    assert log_probs.shape == x.shape
    return log_probs


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(
        beta_start, beta_end, warmup_time, dtype=np.float64
    )
    return betas


def meanflat(x):
    return reduce(x, "b ... -> b", "mean")


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class OriginalDiffusion(torch.nn.Module):
    """
    Contains utilities for the diffusion model.
    Arguments:
    - what the network predicts (x_{t-1}, x_0, or epsilon)
    - which loss function (kl or unweighted MSE)
    - what is the variance of p(x_{t-1}|x_t) (learned, fixed to beta, or fixed to weighted beta)
    - what type of decoder, and how to weight its loss? is its variance learned too?
    """

    def __init__(self, denoise_fn, betas, model_mean_type, model_var_type, loss_type):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.model_mean_type = model_mean_type  # xprev, xstart, eps
        self.model_var_type = model_var_type  # learned, fixedsmall, fixedlarge
        self.loss_type = loss_type  # kl, mse

        assert isinstance(betas, np.ndarray)
        betas = torch.tensor(
            betas.astype(np.float64), dtype=torch.float64
        )  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        buffer("betas", betas)
        buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        buffer(
            "alphas_cumprod_prev", F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        )
        assert self.alphas_cumprod_prev.shape == (timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))
        buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - self.alphas_cumprod))
        buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod))
        buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        buffer(
            "posterior_variance",
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod),
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        buffer(
            "posterior_log_variance_clipped",
            torch.log(
                torch.cat((self.posterior_variance[1:2], self.posterior_variance[1:]))
            ),
        )
        buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod),
        )
        buffer(
            "posterior_mean_coef2",
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod),
        )

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        (bs,) = t.shape
        assert x_shape[0] == bs
        out = a.gather(-1, t)
        assert list(out.shape) == [bs]
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, return_pred_xstart: bool):
        B, C, H, W = x.shape
        assert list(t.shape) == [B], f"t.shape = {t.shape}, B = {B}"
        model_output = self.denoise_fn(x, t)

        # Learned or fixed variance?
        if self.model_var_type == "learned":
            assert model_output.shape == [B, C * 2, H, W]
            model_output, model_log_variance = torch.split(model_output, 2, dim=1)
            model_variance = torch.exp(model_log_variance)
        elif self.model_var_type in ["fixedsmall", "fixedlarge"]:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                "fixedlarge": (
                    self.betas,
                    torch.log(
                        torch.cat((self.posterior_variance[1:2], self.betas[1:]))
                    ),
                ),
                "fixedsmall": (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = self._extract(
                model_variance, t, x.shape
            ) * torch.ones_like(x)
            model_log_variance = self._extract(
                model_log_variance, t, x.shape
            ) * torch.ones_like(x)
        else:
            raise NotImplementedError(self.model_var_type)

        # Mean parameterization
        _maybe_clip = lambda x_: (x_.clamp_(-1.0, 1.0) if clip_denoised else x_)
        if self.model_mean_type == "xprev":  # the model predicts x_{t-1}
            pred_xstart = _maybe_clip(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type == "xstart":  # the model predicts x_0
            pred_xstart = _maybe_clip(model_output)
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        elif self.model_mean_type == "eps":  # the model predicts epsilon
            pred_xstart = _maybe_clip(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, pred_xstart
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            self._extract(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - self._extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    # === Sampling ===

    def p_sample(
        self, x, t, noise_fn, clip_denoised=True, return_pred_xstart: bool = False
    ):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, return_pred_xstart=True
        )
        noise = (
            noise_fn(size=x.shape, dtype=x.dtype, device=x.device)
            if (t > 0).any()
            else 0.0
        )  # no noise if t == 0
        sample = model_mean + torch.exp(0.5 * model_log_variance) * noise
        assert list(sample.shape) == list(
            pred_xstart.shape
        ), f"sample.shape={sample.shape}, pred_xstart.shape={pred_xstart.shape}"
        return (sample, pred_xstart) if return_pred_xstart else sample

    def p_sample_loop(self, shape, noise_fn=torch.randn):
        """
        Generate samples
        """
        assert isinstance(shape, (tuple, list))
        # i_0 = torch.full.constant(self.num_timesteps - 1, dtype=torch.int32)
        img_0 = noise_fn(size=shape, dtype=torch.float32, device=self.betas.device)
        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img_0 = self.p_sample(
                x=img_0,
                t=torch.full([shape[0]], t, device=img_0.device, dtype=torch.int64),
                noise_fn=noise_fn,
                return_pred_xstart=False,
            )

        assert list(img_0.shape) == list(shape)
        return img_0

    def p_sample_loop_progressive(
        self, shape, noise_fn=torch.randn, include_xstartpred_freq=50
    ):
        """
        Generate samples and keep track of prediction of x0
        """
        assert isinstance(shape, (tuple, list))
        # i_0 = tf.constant(self.num_timesteps - 1, dtype=tf.int32)
        img_0 = noise_fn(size=shape, dtype=torch.float32)  # [B, C, H, W]

        # number of x_0 predictions tracked
        num_recorded_xstartpred = self.num_timesteps // include_xstartpred_freq
        xstartpreds_0 = torch.zeros(
            [shape[0], num_recorded_xstartpred, *shape[1:]], dtype=torch.float32
        )  # [B, N, C, H, W]

        for t in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img_0, pred_xstart = self.p_sample(
                x=img_0,
                t=torch.full([shape[0]], t, dtype=torch.int64),
                noise_fn=noise_fn,
                return_pred_xstart=True,
            )
            assert img_0.shape == pred_xstart.shape == shape
            insert_mask = torch.equal(
                torch.floor_divide(t, include_xstartpred_freq),
                torch.arange(num_recorded_xstartpred, dtype=torch.int32),
            )
            insert_mask = torch.reshape(
                insert_mask.to(torch.float32), [1, -1, *[1] * (len(shape) - 1)]
            )
            xstartpreds_0 = (
                insert_mask * pred_xstart[:, None, ...]
                + (1.0 - insert_mask) * xstartpreds_0
            )

        return (
            img_0,
            xstartpreds_0,
        )  # xstart predictions should agree with img_final at step 0

    # === Log likelihood calculation ===

    def _vb_terms_bpd(
        self, x_start, x_t, t, clip_denoised: bool, return_pred_xstart: bool
    ):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            x=x_t, t=t, clip_denoised=clip_denoised, return_pred_xstart=True
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, model_mean, model_log_variance
        )
        kl = meanflat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=model_mean, log_scales=0.5 * model_log_variance
        )
        assert list(decoder_nll.shape) == list(x_start.shape)
        decoder_nll = meanflat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        assert (
            list(kl.shape)
            == list(decoder_nll.shape)
            == list(t.shape)
            == [x_start.shape[0]]
        )
        output = torch.where(t == 0, decoder_nll, kl)
        return (output, pred_xstart) if return_pred_xstart else output

    def training_losses(self, x_start, t, noise=None):
        """
        Training loss calculation
        """

        # Add noise to data
        assert (
            t.shape[0] == x_start.shape[0]
        ), f"t.shape={t.shape[0]}, x_start.shape={x_start.shape[0]}"
        if noise is None:
            noise = torch.randn(
                size=x_start.shape, dtype=x_start.dtype, device=x_start.device
            )
        assert noise.shape == x_start.shape and noise.dtype == x_start.dtype
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Calculate the loss
        if self.loss_type == "kl":  # the variational bound
            losses, pred_xstart = self._vb_terms_bpd(
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                return_pred_xstart=True,
            )
        elif self.loss_type == "mse":  # unweighted MSE
            assert self.model_var_type != "learned"
            target = {
                "xprev": self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[
                    0
                ],
                "xstart": x_start,
                "eps": noise,
            }[self.model_mean_type]
            model_output = self.denoise_fn(x_t, t)
            assert model_output.shape == target.shape == x_start.shape
            losses = meanflat((target - model_output) ** 2)

            pred_xstart = {
                "xprev": self._predict_xstart_from_xprev(x_t, t, model_output),
                "xstart": model_output,
                "eps": self._predict_xstart_from_eps(x_t, t, model_output),
            }[self.model_mean_type]

        else:
            raise NotImplementedError(self.loss_type)

        assert list(losses.shape) == list(
            t.shape
        ), f"losses.shape={losses.shape}, t.shape={t.shape}"
        return losses, pred_xstart

    def _prior_bpd(self, x_start):
        B, T = x_start.shape[0], self.num_timesteps
        qt_mean, _, qt_log_variance = self.q_mean_variance(
            x_start, t=torch.full((B,), T - 1, dtype=torch.int64, device=x_start.device)
        )
        kl_prior = normal_kl(
            mean1=qt_mean,
            logvar1=qt_log_variance,
            mean2=torch.tensor(0.0, device=x_start.device),
            logvar2=torch.tensor(0.0, device=x_start.device),
        )
        assert list(kl_prior.shape) == list(x_start.shape)
        return meanflat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, x_start, clip_denoised=True):
        (B, C, H, W), T = x_start.shape, self.num_timesteps

        bpd_t = torch.zeros([B, T], device=x_start.device)
        mse = torch.zeros([B, T], device=x_start.device)

        for t in reversed(range(0, self.num_timesteps)):
            time_batched = torch.full((B,), t, dtype=torch.int64, device=x_start.device)
            # Calculate VLB term at the current timestep
            new_vlb_b, pred_xstart = self._vb_terms_bpd(
                x_start=x_start,
                x_t=self.q_sample(x_start=x_start, t=time_batched),
                t=time_batched,
                clip_denoised=clip_denoised,
                return_pred_xstart=True,
            )
            # MSE for progressive prediction loss
            assert list(pred_xstart.shape) == list(x_start.shape)
            new_mse_b = meanflat((pred_xstart - x_start) ** 2)
            assert list(new_vlb_b.shape) == list(new_mse_b.shape) == [B]
            # Insert the calculated term into the tensor of all terms
            mask_bt = (
                time_batched[:, None] == torch.arange(T, device=x_start.device)[None, :]
            ).to(torch.float32)
            bpd_t = bpd_t * (1.0 - mask_bt) + new_vlb_b[:, None] * mask_bt
            mse = mse * (1.0 - mask_bt) + new_mse_b[:, None] * mask_bt

        prior_bpd_b = self._prior_bpd(x_start)
        total_bpd_b = torch.sum(bpd_t, dim=1) + prior_bpd_b
        assert list(bpd_t.shape) == list(mse.shape) == [B, T] and list(
            total_bpd_b.shape
        ) == list(prior_bpd_b.shape) == [B]
        return total_bpd_b, bpd_t, prior_bpd_b, mse
