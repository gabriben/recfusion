from sys import exc_info
import numpy as np
import pytest

import recpack.preprocessing.filters as filters
from recpack.preprocessing.preprocessors import (
    DataFramePreprocessor,
    SessionDataFramePreprocessor,
)
from recpack.data.matrix import InteractionMatrix


def test_dataframe_preprocessor_no_filter_no_duplicates(dataframe):

    processor = DataFramePreprocessor(
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
    )

    interaction_m = processor.process(dataframe)

    assert len(processor.item_id_mapping.keys()) == len(
        dataframe[InteractionMatrix.ITEM_IX].unique()
    )
    assert len(processor.user_id_mapping.keys()) == len(
        dataframe[InteractionMatrix.USER_IX].unique()
    )

    assert interaction_m.shape[0] == len(dataframe[InteractionMatrix.USER_IX].unique())
    assert interaction_m.shape[1] == len(dataframe[InteractionMatrix.ITEM_IX].unique())


def test_dataframe_preprocessor_no_filter_duplicates_dedupe(dataframe):
    processor = DataFramePreprocessor(
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
    )

    processor.add_filter(
        filters.Deduplicate(
            InteractionMatrix.ITEM_IX,
            InteractionMatrix.USER_IX,
        )
    )

    org_row = dataframe.loc[3, :].values
    row = org_row.copy()

    row[2] = 12345

    dataframe.loc[dataframe.shape[0]] = row

    interaction_m = processor.process(dataframe)

    assert len(processor.item_id_mapping.keys()) == len(
        dataframe[InteractionMatrix.ITEM_IX].unique()
    )
    assert len(processor.user_id_mapping.keys()) == len(
        dataframe[InteractionMatrix.USER_IX].unique()
    )

    assert interaction_m.shape[0] == len(dataframe[InteractionMatrix.USER_IX].unique())
    assert interaction_m.shape[1] == len(dataframe[InteractionMatrix.ITEM_IX].unique())

    assert org_row[2] == interaction_m.timestamps[row[0], row[1]]
    assert not row[2] == interaction_m.timestamps[row[0], row[1]]


def test_dataframe_preprocessor_no_filter_duplicates_no_dedupe(dataframe):
    processor = DataFramePreprocessor(
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
    )

    org_row = dataframe.loc[3, :].values
    row = org_row.copy()

    row[2] = 12345

    dataframe.loc[dataframe.shape[0]] = row

    interaction_m = processor.process(dataframe)

    assert len(processor.item_id_mapping.keys()) == len(
        dataframe[InteractionMatrix.ITEM_IX].unique()
    )
    assert len(processor.user_id_mapping.keys()) == len(
        dataframe[InteractionMatrix.USER_IX].unique()
    )

    assert interaction_m.shape[0] == len(dataframe[InteractionMatrix.USER_IX].unique())
    assert interaction_m.shape[1] == len(dataframe[InteractionMatrix.ITEM_IX].unique())

    two_values = interaction_m.timestamps[row[0], row[1]]

    assert two_values.shape[0] == 2
    assert row[2] in two_values.values
    assert org_row[2] in two_values.values


def test_dataframe_preprocessor_id_mapping_w_multiple_dataframes(dataframe):
    processor = DataFramePreprocessor(
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        InteractionMatrix.TIMESTAMP_IX,
    )

    dataframe_2 = dataframe.copy()

    org_row = dataframe.loc[3, :].values
    row = org_row.copy()

    row[0] = 666
    row[1] = 333

    dataframe.loc[dataframe.shape[0]] = row

    row_2 = org_row.copy()
    row_2[0] = 111
    row_2[1] = 555
    dataframe_2.loc[dataframe_2.shape[0]] = row_2
    # ---
    interaction_ms = processor.process_many(dataframe, dataframe_2)

    unique_users = set(dataframe[InteractionMatrix.USER_IX].unique()).union(
        dataframe_2[InteractionMatrix.USER_IX].unique()
    )
    unique_items = set(dataframe[InteractionMatrix.ITEM_IX].unique()).union(
        dataframe_2[InteractionMatrix.ITEM_IX].unique()
    )

    assert len(processor.item_id_mapping.keys()) == len(unique_items)
    assert len(processor.user_id_mapping.keys()) == len(unique_users)

    assert len(processor.item_id_mapping.keys()) != len(
        dataframe_2[InteractionMatrix.ITEM_IX].unique()
    )
    assert len(processor.user_id_mapping.keys()) != len(
        dataframe_2[InteractionMatrix.USER_IX].unique()
    )

    assert interaction_ms[0].shape[0] == len(unique_users)
    assert interaction_ms[0].shape[1] == len(unique_items)

    assert interaction_ms[0].shape[0] == interaction_ms[1].shape[0]
    assert interaction_ms[0].shape[1] == interaction_ms[1].shape[1]


def test_raises(dataframe):
    processor = DataFramePreprocessor(
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        InteractionMatrix.TIMESTAMP_IX,
    )

    with pytest.raises(RuntimeError):
        processor._map_users(dataframe)

    with pytest.raises(RuntimeError):
        processor._map_items(dataframe)


def test_dataframe_preprocessor_w_filter_no_duplicates(dataframe):

    processor = DataFramePreprocessor(
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
    )

    myfilter = filters.NMostPopular(
        3,
        InteractionMatrix.ITEM_IX,
    )

    processor.add_filter(myfilter)

    interaction_m = processor.process(dataframe)

    filtered_df = myfilter.apply(dataframe)

    assert len(processor.item_id_mapping.keys()) == len(
        filtered_df[InteractionMatrix.ITEM_IX].unique()
    )
    assert len(processor.user_id_mapping.keys()) == len(
        filtered_df[InteractionMatrix.USER_IX].unique()
    )

    assert interaction_m.shape[0] == len(
        filtered_df[InteractionMatrix.USER_IX].unique()
    )
    assert interaction_m.shape[1] == len(
        filtered_df[InteractionMatrix.ITEM_IX].unique()
    )


def test_add_filter(dataframe):
    processor = DataFramePreprocessor(
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
    )

    processor.add_filter(
        filters.MinUsersPerItem(
            3,
            InteractionMatrix.ITEM_IX,
            InteractionMatrix.USER_IX,
        )
    )
    processor.add_filter(
        filters.NMostPopular(
            3,
            InteractionMatrix.ITEM_IX,
        )
    )

    processor.add_filter(
        filters.MinItemsPerUser(
            3,
            InteractionMatrix.ITEM_IX,
            InteractionMatrix.USER_IX,
        ),
        index=1,
    )

    assert type(processor.filters[0]) == filters.MinUsersPerItem
    assert type(processor.filters[1]) == filters.MinItemsPerUser
    assert type(processor.filters[2]) == filters.NMostPopular


def test_session_dataframe_preprocessor_sunny_day(dataframe_with_fixed_timestamps):
    processor = SessionDataFramePreprocessor(
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
    )
    interaction_m = processor.process(dataframe_with_fixed_timestamps)

    # User 1 has 2 sessions, all other users have a single one.
    assert (
        interaction_m.shape[0]
        == dataframe_with_fixed_timestamps[InteractionMatrix.USER_IX].nunique() + 1
    )

    session_interaction_counts = interaction_m.values.sum(axis=1).T

    session_interaction_counts[0, 0] == 5
    session_interaction_counts[0, 1] == 4
    session_interaction_counts[0, 2] == 3
    session_interaction_counts[0, 3] == 3
    session_interaction_counts[0, 4] == 2
    session_interaction_counts[0, 5] == 3

    # Check that order is as expected
    session_unique_interaction_counts = interaction_m.binary_values.sum(axis=1).T
    session_unique_interaction_counts[0, 2] == 1
    session_unique_interaction_counts[0, 5] == 1

    first_session_id, first_session = next(interaction_m.sorted_item_history)
    assert first_session_id == 0
    np.testing.assert_array_equal(first_session, [0, 1, 2, 3, 4])


@pytest.mark.parametrize(
    "item_ix, user_ix, timestamp_ix",
    [
        (
            "this_is_not_there",
            InteractionMatrix.USER_IX,
            InteractionMatrix.TIMESTAMP_IX,
        ),
        (
            "this_is_not_there",
            InteractionMatrix.USER_IX,
            InteractionMatrix.TIMESTAMP_IX,
        ),  # TODO: add more cases here
    ],
)
def test_session_raises(dataframe, item_ix, user_ix, timestamp_ix):
    processor = SessionDataFramePreprocessor(
        item_ix,
        user_ix,
        timestamp_ix,
    )

    with pytest.raises(ValueError) as excinfo:
        processor.session_transformer(dataframe)
    assert "One of the element doesn't exist!" in str(excinfo.value)


@pytest.mark.parametrize(
    "column_to_drop",
    [
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        InteractionMatrix.TIMESTAMP_IX,
    ],
)
def test_session_raises_missing_column(dataframe, column_to_drop):
    processor = SessionDataFramePreprocessor(
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        InteractionMatrix.TIMESTAMP_IX,
    )
    dataframe.drop(column_to_drop, axis="columns", inplace=True)
    with pytest.raises(ValueError) as excinfo:
        processor.session_transformer(dataframe)
    assert "One of the element doesn't exist!" in str(excinfo.value)
