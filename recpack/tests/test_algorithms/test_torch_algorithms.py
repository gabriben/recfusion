
from unittest.mock import MagicMock
from typing import Callable

import pytest
import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
from torch.autograd import Variable

from recpack.algorithms.torch_algorithms.vaes import (
    MultiVAETorch,
    MultVAE,
    vae_loss_function,
)

# Inspiration for these tests came from:
# https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765

INPUT_SIZE = 1000


@pytest.fixture(scope="function")
def inputs():
    torch.manual_seed(400)
    return Variable(torch.randn(INPUT_SIZE, INPUT_SIZE))


@pytest.fixture(scope="function")
def targets():
    torch.manual_seed(400)
    return Variable(torch.randint(0, 2, (INPUT_SIZE,))).long()


@pytest.fixture(scope="function")
def larger_matrix():
    num_interactions = 2000
    num_users = 500
    num_items = 500

    np.random.seed(400)

    pv_users, pv_items, pv_values = (
        [np.random.randint(0, num_users) for _ in range(0, num_interactions)],
        [np.random.randint(0, num_users) for _ in range(0, num_interactions)],
        [1] * num_interactions,
    )

    pv = scipy.sparse.csr_matrix(
        (pv_values, (pv_users, pv_items)), shape=(num_users + 200, num_items)
    )

    return pv


@pytest.fixture(scope="function")
def mult_vae():
    mult = MultVAE(
        batch_size=500,
        max_epochs=2,
        seed=42,
        learning_rate=1e-2,
        dim_bottleneck_layer=200,
        dim_hidden_layer=600,
        max_beta=0.2,
        anneal_steps=20,
        dropout=0.5,
    )

    mult.save = MagicMock(return_value=True)

    return mult


def assert_changed(params_before, params_after, device):
    # check if variables have changed
    for (_, p0), (_, p1) in zip(params_before, params_after):
        assert not torch.equal(p0.to(device), p1.to(device))


def assert_same(params_before, params_after, device):
    # check if variables have changed
    for (_, p0), (_, p1) in zip(params_before, params_after):
        assert torch.equal(p0.to(device), p1.to(device))


def _training_step(model: nn.Module, loss_fn: Callable, optim: torch.optim.Optimizer, inputs: Variable, targets: Variable, device: torch.device):

    # put model in train mode
    model.train()
    model.to(device)

    #  run one forward + backward step
    # clear gradient
    optim.zero_grad()
    # move data to device
    inputs = inputs.to(device)
    targets = targets.to(device)
    # forward
    likelihood = model(inputs)
    # calc loss
    loss = loss_fn(*likelihood, targets)
    # backward
    loss.backward()
    # optimization step
    optim.step()


def test_training_epoch(mult_vae, larger_matrix):
    mult_vae._init_model(larger_matrix.shape[1])

    users = list(set(larger_matrix.nonzero()[1]))
    params = [np for np in mult_vae.model_.named_parameters() if np[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    # run a training step
    mult_vae._train_epoch(larger_matrix, users)

    device = mult_vae.device

    assert_changed(params_before, params, device)


def test_evaluation_epoch(mult_vae, larger_matrix):
    mult_vae._init_model(larger_matrix.shape[1])

    users = list(set(larger_matrix.nonzero()[1]))
    params = [np for np in mult_vae.model_.named_parameters() if np[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    # TODO Mock mult_vae.save

    # run a training step
    mult_vae._evaluate(larger_matrix, larger_matrix, users)

    device = mult_vae.device

    assert_same(params_before, params, device)


def test_predict(mult_vae, larger_matrix):
    mult_vae._init_model(larger_matrix.shape[1])

    X_pred = mult_vae.predict(larger_matrix)

    assert isinstance(X_pred, scipy.sparse.csr_matrix)

    assert not set(X_pred.nonzero()[0]).difference(larger_matrix.nonzero()[0])


def test_multi_vae_forward(inputs, targets):
    mult_vae = MultiVAETorch(dim_input_layer=INPUT_SIZE)

    params = [np for np in mult_vae.named_parameters() if np[1].requires_grad]

    # take a copy
    params_before = [(name, p.clone()) for (name, p) in params]

    device = torch.device("cpu")

    _training_step(mult_vae, vae_loss_function, torch.optim.Adam(mult_vae.parameters()), inputs, targets, device)
    # do they change after a training step?
    #  let's run a train step and see

    assert_changed(params_before, params, device)
