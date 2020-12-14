from recpack.algorithms.base import Algorithm
import scipy.sparse
import numpy as np
import warnings


def test_check_prediction():
    a = np.ones(5)
    b = a.copy()
    b[2] = 0
    X_pred = scipy.sparse.diags(b).tocsr()
    X = scipy.sparse.diags(a).tocsr()

    a = Algorithm()

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        a._check_prediction(X_pred, X)
        # Verify some things
        assert len(w) == 1

        assert "1 users" in str(w[-1].message)

        a._check_prediction(X, X)
        assert len(w) == 1