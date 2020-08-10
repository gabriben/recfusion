from collections import defaultdict
import logging

import numpy as np
import scipy.sparse


logger = logging.getLogger("recpack")


def groupby2(keys, values):
    """ A group by of separate lists where order doesn't matter. """
    multidict = defaultdict(list)
    for k, v in zip(keys, values):
        multidict[k].append(v)
    return multidict.items()


def to_tuple(el):
    """ Whether single element or tuple, always returns as tuple. """
    if type(el) == tuple:
        return el
    else:
        return (el,)


def df_to_sparse(df, item_ix, user_ix, value_ix=None, shape=None):
    if value_ix is not None and value_ix in df:
        values = df[value_ix]
    else:
        if value_ix is not None:
            # value_ix provided, but not in df
            logger.warning(
                f"Value column {value_ix} not found in dataframe. Using ones instead."
            )

        num_entries = df.shape[0]
        # Scipy sums up the entries when an index-pair occurs more than once,
        # resulting in the actual counts being stored. Neat!
        values = np.ones(num_entries)

    indices = list(zip(*df.loc[:, [user_ix, item_ix]].values))

    if indices == []:
        indices = [[], []]  # Empty zip does not evaluate right

    if shape is None:
        shape = df[user_ix].max() + 1, df[item_ix].max() + 1
    sparse_matrix = scipy.sparse.csr_matrix(
        (values, indices), shape=shape, dtype=values.dtype
    )

    return sparse_matrix
