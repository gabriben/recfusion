import logging

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.lil import lil_matrix
from torch._C import device
from tqdm.auto import tqdm

import torch

from recpack.algorithms import Algorithm
from recpack.algorithms.util import naive_sparse2tensor, get_batches, get_users
from recpack.data.matrix import to_binary

logger = logging.getLogger("recpack")


class WeightedMatrixFactorization(Algorithm):
    """WMF Algorithm by Yifan Hu, Yehuda Koren and Chris Volinsky et al.

    As described in Hu, Yifan, Yehuda Koren, and Chris Volinsky.
    "Collaborative filtering for implicit feedback datasets."
    2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008

    Based on the input data a confidence of the interaction is computed.
    Parametrized by alpha and epsilon (hyper parameters)

    - If the chosen confidence scheme is ``'minimal'``,
      confidence is computed as ``c(u,i) = 1 + alpha * r(u,i)``.
    - If the chosen confidence scheme is ``'log-scaling'``,
      confidence is computed as ``c(u,i) = 1 + alpha * log(1 + r(u,i)/epsilon)``

    Since the data during fitting is assumed to be implicit,
    this confidence will be the same for all interactions,
    and as such leaving the HP to the defaults works good enough.

    **Example of use**::

        import numpy as np
        from scipy.sparse import csr_matrix
        from recpack.algorithms import WMF

        X = csr_matrix(np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]]))

        algo = WMF()
        # Fit algorithm
        algo.fit(X)

        # Get the predictions
        predictions = algo.predict(X)

        # Predictions is a csr matrix, inspecting the scores with
        predictions.toarray()

    :param conficence_scheme: Which confidence scheme should be used
        to calculate the confidence matrix.
        Options are ["minimal", "log-scaling"].
        Defaults to "minimal"
    :type conficence_scheme: string, optional
    :param alpha: Scaling parameter for generating confidences from ratings.
        Defaults to 40.
    :type alpha: int, optional
    :param epsilon: Small value to avoid division by zero,
        used to compute a confidence from a rating.
        Only used in case cs is set to 'log-scaling'
        Defaults to 1e-8
    :type epsilon: float, optional
    :param num_components: Dimension of the embeddings of both user- and item-factors.
        Defaults to 100
    :type num_components: int, optional
    :param regularization: Regularization parameter used to calculate the Least Squares.
        Defaults to 0.01
    :type regularization: float, optional
    :param iterations: Number of iterations to execute the ALS calculations.
        Defaults to 20
    :type iterations: int, optional
    """

    CONFIDENCE_SCHEMES = ["minimal", "log-scaling"]
    """Allowed values for confidence scheme parameter"""

    def __init__(
        self,
        confidence_scheme: str = "minimal",
        alpha: int = 40,
        epsilon: float = 1e-8,
        num_components: int = 100,
        regularization: float = 0.01,
        iterations: int = 20,
    ):
        """
        Initialize the weighted matrix factorization algorithm
        with confidence generator parameters.
        """
        super().__init__()
        self.confidence_scheme = confidence_scheme
        if confidence_scheme in self.CONFIDENCE_SCHEMES:
            self.confidence_scheme = confidence_scheme
        else:
            raise ValueError("Invalid confidence scheme parameter.")

        self.alpha = alpha
        self.epsilon = epsilon

        self.num_components = num_components
        self.regularization = regularization
        self.iterations = iterations

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

    def _fit(self, X: csr_matrix) -> None:
        """Calculate the user- and item-factors which will approximate X
            after applying a dot-product.

        :param X: Sparse user-item matrix which will be used to fit the algorithm.
        """
        self.num_users, self.num_items = X.shape
        # Create a matrix with only nonzero users and items.
        self.user_factors_, self.item_factors_ = self._alternating_least_squares(
            X)

    def _predict(self, X: csr_matrix) -> csr_matrix:
        """Prediction scores are calculated as the dotproduct of
            the recomputed user-factors and the item-factors.

        :param X: Sparse user-item matrix which will be used to do the predictions;
            only for set of users with interactions will recommendations be generated.
        :return: User-item matrix with the prediction scores as values.
        """

        U_conf = self._generate_confidence(X)
        U_user_factors = self._least_squares(
            U_conf, self.item_factors_
        ).detach().cpu().numpy()

        score_matrix = csr_matrix(
            U_user_factors @ self.item_factors_.detach().cpu().numpy().T)

        self._check_prediction(score_matrix, X)
        return score_matrix

    def _generate_confidence(self, r) -> csr_matrix:
        """
        Generate the confidence matrix as described in the paper.
        This can be calculated in different ways:
          - Minimal: c_ui = alpha * r_ui
          - Log scaling: c_ui = alpha * log(1 + r_ui / epsilon)
        NOTE: This implementation deviates from the paper.
        The additional +1 won't be stored to keep the confidence matrix sparse.
        For this reason C-1 will be the result of this function.
        Important is that it will impact the least squares calculation.
        :param r: User-item matrix which the calculations are based on.
        :return: User-item matrix converted with the confidence values.
        """
        result = csr_matrix(r, copy=True)
        if self.confidence_scheme == "minimal":
            result.data = self.alpha * result.data
        elif self.confidence_scheme == "log-scaling":
            result.data = self.alpha * np.log(1 + result.data / self.epsilon)

        return result

    def _alternating_least_squares(self, X: csr_matrix) -> (np.ndarray, np.ndarray):
        """
        The ALS algorithm will execute the least squares calculation for x number of iterations.
        According factorizing matrix C into two factors Users and Items such that R \approx U^T I.
        :param X: Sparse matrix which the ALS algorithm should be applied on.
        :return: Generated user- and item-factors based on the input matrix X.
        """

        # user_factors = (
        #     np.random.rand(
        #         self.num_users, self.num_components).astype(np.float32)
        #     * 0.01
        # )
        # item_factors = (
        #     np.random.rand(
        #         self.num_items, self.num_components).astype(np.float32)
        #     * 0.01
        # )

        C = self._generate_confidence(X)
        # c_torch = naive_sparse2tensor(c)

        item_factors = torch.rand(
            (self.num_items, self.num_components), dtype=torch.float32, device=self.device) * 0.01

        for i in tqdm(range(self.iterations * 2)):

            if i % 2 == 0:
                # User iteration
                user_factors = self._least_squares(
                    C, item_factors
                )
            else:
                # Item iteration
                item_factors = self._least_squares(
                    C.T, user_factors
                )

            # old_uf = np.array(user_factors, copy=True)
            # old_if = np.array(item_factors, copy=True)

            # norm_uf = np.linalg.norm(old_uf - user_factors, "fro")
            # norm_if = np.linalg.norm(old_if - item_factors, "fro")
            # logger.debug(
            #     f"{self.name} - Iteration {i} - L2-norm of diff user_factors: {norm_uf} - L2-norm of diff "
            #     f"item_factors: {norm_if}"
            # )

        return user_factors, item_factors

    def _least_squares(
        self,
        C: csr_matrix,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the other factor based on the confidence matrix and the factors with the least squares algorithm.
        It is a general function for item- and user-factors. Depending on the parameter factor_type the other factor
        will be calculated.
        :param conf_matrix: (Transposed) Confidence matrix
        :return: Other factor nd-array based on the factor array and the confidence matrix
        """
        # factors_x = np.zeros((dimension, self.num_components))
        YtY = Y.T @ Y

        # accumulate YtCxY + regularization * I in A
        # -----------
        # Because of the impact of calculating C-1, instead of C,
        # calculating YtCxY is a bottleneck, so the significant speedup calculations will be used:
        #  YtCxY = YtY + Yt(Cx)Y
        # Left side of the linear equation A will be:
        #  A = YtY + Yt(Cx)Y + regularization * I
        #  For each x, let us define the diagonal n × n matrix Cx where Cx_yy = c_xy

        binary_C = to_binary(C)

        factors = torch.zeros(Y.shape)

        for id_batch in get_batches(get_users(C), batch_size=100):
            # Create batches of 100 at the same time
            C_diag_batch = torch.diag_embed(torch.Tensor(
                C[id_batch, :].toarray())).to(self.device)

            # A batch needs to be a tensor.
            A_batch = YtY + (Y.T @ C_diag_batch) @ Y + self.regularization * \
                torch.eye(self.num_components, device=self.device)

            P_batch = naive_sparse2tensor(binary_C[id_batch, :]).unsqueeze(-1)

            B_batch = (
                Y.T @ (C_diag_batch + torch.eye(C_diag_batch.shape[1], device=self.device))) @ P_batch

            # Accumulate Yt(Cx + I)Px in b
            # Solve the problem with the A_batch, save results.
            # Xu = (YtCxY + regularization * I)^-1 (YtCxPx)
            x_batch = torch.linalg.lstsq(A_batch, B_batch).solution.squeeze(-1)

            factors[id_batch] = x_batch

        return factors
