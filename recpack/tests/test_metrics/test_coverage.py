import numpy as np
from recpack.metrics.coverage import CoverageK


def test_coverageK(X_pred, X_true):
    K = 2
    metric = CoverageK(K)

    metric.calculate(X_true, X_pred)

    # user 0 gets recommended items 0 and 2
    # user 2 gets recommended items 3 and 4
    # total number of items = 5
    np.testing.assert_almost_equal(metric.value, 4 / 5)
