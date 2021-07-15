import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from recpack.data.matrix import InteractionMatrix

INPUT_SIZE = 1000
USER_IX = InteractionMatrix.USER_IX
ITEM_IX = InteractionMatrix.ITEM_IX
TIMESTAMP_IX = InteractionMatrix.TIMESTAMP_IX


@pytest.fixture(scope="function")
def data():
    input_dict = {
        InteractionMatrix.USER_IX: [2, 2, 2, 0, 0, 0],
        InteractionMatrix.ITEM_IX: [1, 3, 4, 0, 2, 4],
        "values": [1, 2, 1, 1, 1, 2],
    }

    matrix = scipy.sparse.csr_matrix(
        (
            input_dict["values"],
            (
                input_dict[InteractionMatrix.USER_IX],
                input_dict[InteractionMatrix.ITEM_IX],
            ),
        ),
        shape=(10, 5),
    )
    return matrix


@pytest.fixture(scope="function")
def ranked_data_complete():
    ranked_users = [0, 0, 0, 2, 2, 2]
    ranked_items = [0, 2, 4, 1, 3, 4]
    ranked_ranks = [3, 2, 1, 3, 1, 2]

    matrix = scipy.sparse.csr_matrix(
        (ranked_ranks, (ranked_users, ranked_items)),
        shape=(10, 5),
    )
    return matrix


@pytest.fixture(scope="function")
def data_knn():
    pred_users, pred_items, pred_values = (
        [0, 0, 0, 2, 2, 2],
        [0, 2, 3, 1, 3, 4],
        [0.3, 0.2, 0.1, 0.23, 0.3, 0.5],
    )

    pred = scipy.sparse.csr_matrix(
        (pred_values, (pred_users, pred_items)), shape=(10, 5)
    )

    return pred


@pytest.fixture(scope="function")
def mat():
    data = {
        TIMESTAMP_IX: [3, 2, 1, 4, 0, 1, 2, 4, 0, 1, 2],
        ITEM_IX: [0, 1, 2, 3, 0, 1, 2, 4, 0, 1, 2],
        USER_IX: [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
    }
    df = pd.DataFrame.from_dict(data)

    return InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)


@pytest.fixture(scope="function")
def larger_mat():
    data = {
        TIMESTAMP_IX: np.random.randint(0, 100, size=100),
        ITEM_IX: np.random.randint(0, 25, size=100),
        USER_IX: np.random.randint(0, 100, size=100),
    }
    df = pd.DataFrame.from_dict(data)

    return InteractionMatrix(df, ITEM_IX, USER_IX, timestamp_ix=TIMESTAMP_IX)
