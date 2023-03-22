import pytest

from code.dataset import HouseDataset

INITIAL_COLUMNS = 81
USEFUL_COLUMNS = 78  # dopo aver rimosso le colonne con troppi valori nulli
FEATURES_WITH_ID = 77
CATEGORICAL_RAW = 43
CATEGORICAL = 0
NUMERICAL_RAW = INITIAL_COLUMNS - CATEGORICAL_RAW


def test_dataset():
    dataset = HouseDataset()
    # controllo numeri di colonne, features e target
    # raw=True significa prima del preprocessing
    assert dataset.get_n_features_with_id() == FEATURES_WITH_ID
    assert dataset.get_n_columns(raw=True) == INITIAL_COLUMNS
    assert dataset.get_n_columns() == USEFUL_COLUMNS
    assert len(dataset.get_categorical()) == CATEGORICAL_RAW
    assert len(dataset.get_categorical(raw=False)) == CATEGORICAL
    assert len(dataset.get_numerical(raw=True, include_id=True)) == NUMERICAL_RAW
    assert len(dataset.get_numerical(raw=False, include_id=True)) == USEFUL_COLUMNS
    assert len(dataset.get_numerical(raw=True, include_id=False)) == NUMERICAL_RAW - 1
    assert len(dataset.get_numerical()) == USEFUL_COLUMNS - 1
    assert len(dataset.get_numerical()) + len(dataset.get_categorical(raw=False)) == dataset.get_n_features_with_id()
    assert len(dataset.get_numerical(raw=True)) + len(dataset.get_categorical(raw=True)) == dataset.get_n_features_with_id(raw=True)

    # verifico che il numero di esempi target sia pari al numero di esempi features
    assert dataset.get_target().size == dataset.get_features_with_separated_id()[1].iloc[:, 0].size


def test_ordinal_encoding():
    ds = HouseDataset()
    ds.__ordinal_encoding()
