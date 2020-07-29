import numpy as np

from recpack.metricsv2.dcg import DCG, DCGK

def test_dcgk_simple(X_pred, X_true_simplified):
    K = 2
    metric = DCGK(K)

    metric.update(X_pred, X_true_simplified)

    expected_value = (
        (1 / np.log2(2 + 1)) +    # rank 1
        1                         # rank 0
    ) / 2  # 2 users

    assert metric.name == "DCG_2"
    assert metric.results.user.nunique() == 2
    np.testing.assert_almost_equal(metric.value, expected_value)

def test_dcg_simple(X_pred, X_true_simplified):
    metric = DCG()

    metric.update(X_pred, X_true_simplified)

    expected_value = (
        (1 / np.log2(2 + 1)) +    # rank 1
        1                         # rank 0
    ) / 2  # 2 users
    
    assert metric.name == "DCG"
    assert metric.results.user.nunique() == 2
    np.testing.assert_almost_equal(metric.value, expected_value)


def test_dcgk(X_pred, X_true):
    K = 2
    metric = DCGK(K)

    metric.update(X_pred, X_true)

    # user 0 has 2 correct items, user 2 has two correct items

    expected_value = (
            (1 / np.log2(2 + 1))  +       # user 0 rank 1
            (
                1 +                       # user 2 rank 0
                (1 / np.log2(2 + 1))      # user 2 rank 1
            )
     ) / 2  # 2 users

    assert metric.name == "DCG_2"
    assert metric.results.user.nunique() == 2
    np.testing.assert_almost_equal(metric.value, expected_value)

def test_dcg(X_pred, X_true):

    metric = DCG()

    metric.update(X_pred, X_true)

    # user 0 has 2 correct items, user 2 has 2 correct items

    expected_value = (
            (
                1 / np.log2(2 + 1) +             # user 2 rank 1
                1 / np.log2(3 + 1)               # user 2 rank 2
            )  +       
            (
                1 +                              # user 0 rank 0
                (1 / np.log2(2 + 1))             # user 0 rank 1
            )
     ) / 2  # 2 users

    assert metric.name == "DCG"
    assert metric.results.user.nunique() == 2
    np.testing.assert_almost_equal(metric.value, expected_value)

def test_dcgk_3(X_pred, X_true):
    K = 3
    metric = DCGK(K)

    metric.update(X_pred, X_true)

    # user 0 has 2 correct items, user 2 has 2 correct items
    expected_value = (
            (
                1 / np.log2(2 + 1) +             # user 2 rank 1
                1 / np.log2(3 + 1)               # user 2 rank 2
            )  +       
            (
                1 +                              # user 0 rank 0
                (1 / np.log2(2 + 1))             # user 0 rank 1
            )
     ) / 2  # 2 users

    assert metric.name == "DCG_3"
    assert metric.results.user.nunique() == 2
    np.testing.assert_almost_equal(metric.value, expected_value)
