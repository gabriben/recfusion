from typing import List
from abc import ABC, abstractmethod

import pandas as pd

# TODO Improve interface so that arguments known to the Preprocessor don't
# have to be duplidated.


class Filter(ABC):
    def __init__(self, item_id, user_id, timestamp_id=None):
        self.user_id = user_id
        self.item_id = item_id
        self.timestamp_id = timestamp_id

    def apply_all(self, *dfs: pd.DataFrame) -> List[pd.DataFrame]:
        ret = list()
        for df in dfs:
            ret.append(self.apply(df))
        return ret

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Filter to the DataFrame passed.

        :param df: DataFrame to filter
        :type df: pd.DataFrame
        """
        raise NotImplementedError

    def __str__(self):
        attrs = self.__dict__.copy()
        for k in ["user_id", "item_id", "timestamp_id"]:
            del attrs[k]
        return f"{self.__class__.__name__}({', '.join((f'{k}={v}' for k, v in attrs.items()))})"


class MinUsersPerItem(Filter):
    def __init__(
        self,
        min_users_per_item: int,
        item_id: str,
        user_id: str,
        timestamp_id: str = None,
        count_duplicates: bool = False,
    ):
        """
        Require that a minimum number of users has interacted with an item.

        :param min_users_per_item: Minimum number of users required.
        :type min_users_per_item: int
        :param item_id: Name of the column in which item identifiers are listed.
        :type item_id: str
        :param user_id: Name of the column in which user identifiers are listed.
        :type user_id: str
        :param timestamp_id: Name of the column in which timestamps are listed, defaults to None
        :type timestamp_id: str, optional
        :param count_duplicates: Count multiple interactions with the same user, defaults to False
        :type count_duplicates: bool
        """
        self.min_ui = min_users_per_item
        self.count_duplicates = count_duplicates

        super().__init__(item_id, user_id, timestamp_id=timestamp_id)

    def apply(self, df):
        iids = (
            df[self.item_id]
            if self.count_duplicates
            else df.drop_duplicates([self.user_id, self.item_id])[self.item_id]
        )
        cnt_users_per_item = iids.value_counts()
        items_of_interest = list(
            cnt_users_per_item[cnt_users_per_item >= self.min_ui].index
        )

        return df[df[self.item_id].isin(items_of_interest)]


class NMostPopular(Filter):
    def __init__(self, N: int, item_id: str, user_id: str, timestamp_id=None):
        """
        Retain only the N most popular items.
        Note: All interactions count, also if a user interacted with the same item
        1000 times.

        :param N: Number of items to retain.
        :type N: int
        :param item_id: Name of the column in which item identifiers are listed.
        :type item_id: str
        :param user_id: Name of the column in which user identifiers are listed.
        :type user_id: str
        :param timestamp_id: Name of the column in which timestamps are listed, defaults to None
        :type timestamp_id: str, optional
        """
        self.N = N

        super().__init__(item_id, user_id, timestamp_id=timestamp_id)

    def apply(self, df):
        cnt_users_per_item = df[self.item_id].value_counts(sort=True, ascending=False)

        items_of_interest = list(cnt_users_per_item[0 : self.N].index)

        return df[df[self.item_id].isin(items_of_interest)]


class MinItemsPerUser(Filter):
    def __init__(
        self,
        min_items_per_user: int,
        item_id: str,
        user_id: str,
        timestamp_id: str = None,
        count_duplicates: bool = False,
    ):
        """
        Require that a user has interacted with a minimum number of items.

        :param min_items_per_user: Minimum number of items required.
        :type min_items_per_user: int
        :param item_id: Name of the column in which item identifiers are listed.
        :type item_id: str
        :param user_id: Name of the column in which user identifiers are listed.
        :type user_id: str
        :param timestamp_id: Name of the column in which timestamps are listed, defaults to None
        :type timestamp_id: str, optional
        :param count_duplicates: Count multiple interactions with the same item, defaults to False
        :type count_duplicates: bool
        """
        self.min_iu = min_items_per_user
        self.count_duplicates = count_duplicates

        super().__init__(item_id, user_id, timestamp_id=timestamp_id)

    def apply(self, df):
        uids = (
            df[self.user_id]
            if self.count_duplicates
            else df.drop_duplicates([self.user_id, self.item_id])[self.user_id]
        )
        cnt_items_per_user = uids.value_counts()
        users_of_interest = list(
            cnt_items_per_user[cnt_items_per_user >= self.min_iu].index
        )

        return df[df[self.user_id].isin(users_of_interest)]
