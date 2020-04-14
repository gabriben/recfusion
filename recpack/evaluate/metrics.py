import enum
import numpy
import scipy.sparse


class Metric:
    pass


class RecallK(Metric):
    def __init__(self, K):
        self.K = K
        self.recall = 0
        self.num_users = 0

    def update(self, X_pred, X_true, users=None):
        # resolve top K items per user
        # Get indices of top K items per user

        # Per user get a set of the topK predicted items
        topK_items_sets = {
            u: set(best_items_row[-self.K:])
            for u, best_items_row in enumerate(numpy.argpartition(X_pred, -self.K))
        }

        # Per user get a set of interacted items.
        items_sets = {u: set(X_true[u].nonzero()[1]) for u in range(X_true.shape[0])}

        for u in topK_items_sets.keys():
            recommended_items = topK_items_sets[u]
            true_items = items_sets[u]

            self.recall += len(recommended_items.intersection(true_items)) / min(
                self.K, len(true_items)
            )
            self.num_users += 1

        return

    @property
    def value(self):
        if self.num_users == 0:
            return 0

        return self.recall / self.num_users


class MeanReciprocalRankK(Metric):
    def __init__(self, K):
        self.K = K
        self.rr = 0
        self.num_users = 0

    def update(self, X_pred, X_true, users=None):
        # Per user get a sorted list of the topK predicted items
        topK_items = {
            u: best_items_row[-self.K:][
                numpy.argsort(X_pred[u][best_items_row[-self.K:]])
            ][::-1]
            for u, best_items_row in enumerate(numpy.argpartition(X_pred, -self.K))
        }

        items_sets = {u: set(X_true[u].nonzero()[1]) for u in range(X_true.shape[0])}

        for u in topK_items.keys():
            for ix, item in enumerate(topK_items[u]):
                if item in items_sets[u]:
                    self.rr += 1 / (ix + 1)
                    break

            self.num_users += 1

        return

    @property
    def value(self):
        if self.num_users == 0:
            return 0

        return self.rr / self.num_users


class NDCGK(Metric):
    def __init__(self, K):
        self.K = K
        self.NDCG = 0
        self.num_users = 0

        self.discount_template = 1.0 / numpy.log2(numpy.arange(2, K + 2))

        self.IDCG = {K: self.discount_template[:K].sum()}

    def update(self, X_pred, X_true, users=None):

        topK_items = {
            u: best_items_row[-self.K:][
                numpy.argsort(X_pred[u][best_items_row[-self.K:]])
            ][::-1]
            for u, best_items_row in enumerate(numpy.argpartition(X_pred, -self.K))
        }

        items_sets = {u: set(X_true[u].nonzero()[1]) for u in range(X_true.shape[0])}

        for u in topK_items.keys():
            M = len(items_sets[u])
            if M < self.K:
                IDCG = self.IDCG.get(M)

                if not IDCG:
                    IDCG = self.discount_template[:M].sum()
                    self.IDCG[M] = IDCG
            else:
                IDCG = self.IDCG[self.K]

            # Compute DCG
            DCG = sum(
                (self.discount_template[rank] * (item in items_sets[u]))
                for rank, item in enumerate(topK_items[u])
            )

            self.num_users += 1
            self.NDCG += DCG / IDCG

        return

    @property
    def value(self):
        if self.num_users == 0:
            return 0

        return self.NDCG / self.num_users


class PropensityType(enum.Enum):
    """
    Propensity Type enum.
    - Uniform means each item has the same propensity
    - Global means each item has a single propensity for all users
    - User means that for each user, item combination a unique propensity is computed.
    """

    UNIFORM = 1
    GLOBAL = 2
    USER = 3


class InversePropensity:
    """
    Abstract inverse propensity class
    """

    def __init__(self, data_matrix):
        self.data_matrix = data_matrix

    def get(self, users):
        pass


class SNIPS(Metric):
    def __init__(self, K: int, inverse_propensities: InversePropensity):
        self.K = K
        self.num_users = 0
        self.score = 0

        self.inverse_propensities = inverse_propensities

    def update(
        self, X_pred: numpy.matrix, X_true: scipy.sparse.csr_matrix, users: list
    ) -> None:
        """
        Update the internal metrics given a set of recommendations and actual events.
        """
        assert X_pred.shape == X_true.shape
        if self.inverse_propensities is None:
            raise Exception(
                "inverse_propensities should not be None, fit a propensity model before using the SNIPS metric"
            )

        # Top K mask on the X_pred
        topK_item_sets = {
            u: set(best_items_row[-self.K:])
            for u, best_items_row in enumerate(numpy.argpartition(X_pred, -self.K))
        }

        user_indices, item_indices = zip(
            *[(u, i) for u in topK_item_sets for i in topK_item_sets[u]]
        )
        values = [1 for i in range(len(user_indices))]
        topK_mask = scipy.sparse.csr_matrix(
            (values, (user_indices, item_indices)), shape=X_pred.shape
        )

        x_pred_top_k = scipy.sparse.csr_matrix(X_pred, shape=X_pred.shape).multiply(
            topK_mask
        )
        # binarize the prediction matrix
        x_pred_top_k[x_pred_top_k > 0] = 1

        ip = self.inverse_propensities.get(users)

        X_pred_as_propensity = x_pred_top_k.multiply(ip).tocsr()
        X_true_as_propensity = X_true.multiply(ip).tocsr()

        X_hit_inverse_propensities = X_pred_as_propensity.multiply(X_true)

        for user in range(X_pred.shape[0]):
            # Compute the hit score as the sum of propensities of hit items.
            # (0 contribution to the score if the item is not in X_true)
            hit_score = X_hit_inverse_propensities[user].sum()

            # Compute the max possible hit score, as the sum of the propensities of recommended items.
            max_possible_hit_score = X_true_as_propensity[user].sum()

            # Compute the user's actual number of interactions to normalise the metric.
            # And take min with self.K since it will be impossible to hit all of the items if there are more than K
            number_true_interactions = X_true[user].sum()

            if number_true_interactions > 0 and max_possible_hit_score > 0:
                self.num_users += 1
                self.score += (1 / max_possible_hit_score) * hit_score

    @property
    def value(self):
        if self.num_users == 0:
            return 0

        return self.score / self.num_users

    @property
    def name(self):
        return f"SNIPS@{self.K}"


IP_CAP = 10000


class UniformInversePropensity:
    """
    Helper class which will do the uniform propensity computation up front
    and return the results
    """

    def __init__(self, data_matrix):
        self.inverse_propensities = 1 / self._get_propensities(data_matrix)
        self.inverse_propensities[self.inverse_propensities > IP_CAP] = IP_CAP

    def get(self, users):
        return self.inverse_propensities

    def _get_propensities(self, data_matrix):
        nr_items = data_matrix.shape[1]
        return numpy.array([[1 / nr_items for i in range(nr_items)]])


class GlobalInversePropensity(InversePropensity):
    """
    Class to compute global propensities and serve them
    """

    def __init__(self, data_matrix):
        self.inverse_propensities = 1 / self._get_propensities(data_matrix)
        self.inverse_propensities[self.inverse_propensities > IP_CAP] = IP_CAP

    def get(self, users):
        return self.inverse_propensities

    def _get_propensities(self, data_matrix):
        item_count = data_matrix.sum(axis=0)
        total = data_matrix.sum()
        propensities = item_count / total
        propensities[propensities == numpy.inf] = 0
        return propensities


class UserInversePropensity(InversePropensity):
    """
    Class to compute propensities on the fly for a set of users.
    """

    def __init__(self, data_matrix):
        self.data_matrix = data_matrix

    def get(self, users):
        # Compute the inverse propensities on the fly
        propensities = self._get_propensities(users)
        inverse_propensities = propensities.copy()
        inverse_propensities.data = 1 / propensities.data
        # Cap the inverse propensity to sensible values,
        # otherwise we will run into division by almost 0 issues.
        inverse_propensities[inverse_propensities > IP_CAP] = IP_CAP
        return inverse_propensities

    def _get_propensities(self, users):
        row_sum = self.data_matrix[users].sum(axis=1)
        propensities = scipy.sparse.csr_matrix(self.data_matrix[users] / row_sum)
        return propensities


class SNIPSFactory:
    def __init__(self, propensity_type):
        self.propensity_type = propensity_type
        self.inverse_propensities = None

    def fit(self, data):
        """
        Fit the propensities based on a sparse matrix with recommendation counts
        """
        if self.propensity_type == PropensityType.UNIFORM:
            self.inverse_propensities = UniformInversePropensity(data)
        elif self.propensity_type == PropensityType.GLOBAL:
            self.inverse_propensities = GlobalInversePropensity(data)
        elif self.propensity_type == PropensityType.USER:
            self.inverse_propensities = UserInversePropensity(data)
        else:
            raise ValueError(f"Unknown propensity type {self.propensity_type}")

    def create_multipe_SNIPS(self, K_values: list) -> dict:
        snipses = {}
        for K in K_values:
            snipses[f"SNIPS@{K}"] = self.create(K)
        return snipses

    def create(self, K: int) -> SNIPS:
        if self.inverse_propensities is None:
            raise Exception(
                "inverse_propensities should not be None, fit a propensity model before creating SNIPS metrics"
            )
        return SNIPS(K, self.inverse_propensities)


class MutexMetric:
    # TODO Refactor into a Precision metric, a FP and a TN metric. 
    """
    Metric used to evaluate the mutex predictors

    Computes false positives as predictor says it's mutex, but sample shows that the item predicted as mutex has been purchased in the evaluation data
    Computes True negatives as items predicted as non mutex, which are also actually in the evaluation data.
    """

    def __init__(self):
        self.false_positives = 0
        self.positives = 0

        self.true_negatives = 0
        self.negatives = 0

    def update(
        self, X_pred: numpy.matrix, X_true: scipy.sparse.csr_matrix, users: list
    ) -> None:

        self.positives += X_pred.sum()
        self.negatives += (X_pred.shape[0] * X_pred.shape[1]) - X_pred.sum()

        false_pos = scipy.sparse.csr_matrix(X_pred).multiply(X_true)
        self.false_positives += false_pos.sum()

        negatives = numpy.ones(X_pred.shape) - X_pred
        true_neg = scipy.sparse.csr_matrix(negatives).multiply(X_true)
        self.true_negatives += true_neg.sum()

    @property
    def value(self):
        return (
            self.false_positives,
            self.positives,
            self.true_negatives,
            self.negatives,
        )
