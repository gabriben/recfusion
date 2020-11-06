from unittest.mock import MagicMock

import pytest
import numpy as np
import scipy.sparse
import torch
from torch.autograd import Variable

from recpack.algorithms import RecVAE, MultVAE
# Inspiration for these tests came from:
# https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765


INPUT_SIZE = 1000


@pytest.fixture(scope="function")
def input_size():
    return INPUT_SIZE


@pytest.fixture(scope="function")
def inputs():
    torch.manual_seed(400)
    return Variable(torch.randn(INPUT_SIZE, INPUT_SIZE))


@pytest.fixture(scope="function")
def targets():
    torch.manual_seed(400)
    return Variable(torch.randint(0, 2, (INPUT_SIZE,))).long()


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


@pytest.fixture(scope="function")
def rec_vae():
    rec = RecVAE(
        batch_size=500,
        max_epochs=2,
        seed=42,
        learning_rate=1e-2,
        dim_bottleneck_layer=200,
        dim_hidden_layer=600,
        dropout=0.5,
    )

    rec.save = MagicMock(return_value=True)

    return rec
