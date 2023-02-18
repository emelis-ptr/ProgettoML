import pandas as pd
from pandas import DataFrame, Series
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from dataset import HouseDataset

from enum import Enum
from abc import ABC, abstractmethod


class FeaturesSelection(ABC):
    """
    Interfaccia che definisce un singolo metodo per le feature selections
    """

    def __init__(self):
        self.name: str = ""

    @abstractmethod
    def select_features(self, k: int, x_train: DataFrame, y_train: Series) -> (DataFrame, Series):
        """
        Esegue la feature selection su un dataset
        :param x_train: il dataset di training
        :param y_train: il dataset di testing
        :param k: il numero di feature da selezionare
        """
        pass


class SelectKBestFS(FeaturesSelection):

    def __init__(self):
        super().__init__()
        self.name = "_SelectKBest"

    def select_features(self, k: int, x_train: DataFrame, y_train: Series) -> (DataFrame, Series):
        # seleziona (K) top features
        selector = SelectKBest(f_regression, k=k)
        x_new = selector.fit_transform(x_train, y_train)

        # ottengo un vettore di 0,1 dove 1 indica i nomi delle colonne selezionate
        # mask = selector.get_support()
        # accedo ai nomi delle colonne
        # selected_features = x.columns[mask]

        # crea un nuovo dataframe con le colonne selezionate
        # x_new_df = pd.DataFrame(x_new, columns=selected_features)

        # converte data in tensor
        # x_new_tensor: Tensor = torch.from_numpy(x_new).float()
        # y_tensor: Tensor = torch.from_numpy(y.values).float()
        # TODO convertire in Dataframe??? Oppure lasciare tensor?
        return pd.DataFrame(x_new), y_train


class MutualInformationFS(FeaturesSelection):

    def __init__(self):
        super().__init__()
        self.name = "_MutualInformation"

    def select_features(self, k: int, x_train: DataFrame, y_train: Series) -> (DataFrame, Series):
        """
        La quantità d'informazioni che ogni feature da rispetto alle altre.
        :return: Le features selezionate
        """
        # calcola la informazione mutua tra le features rispetto al valore target
        mi = mutual_info_regression(x_train, y_train)
        # crea il dataframe dei risultati con la mutua informazione
        dmi = pd.DataFrame(mi, index=x_train.columns, columns=['mi']).sort_values(by='mi', ascending=False)
        # prende dal datagrame le k feature con più informazione mutua
        feat = list(dmi.index[:k])
        # restituisco dal training solo le k feature con più informazione mutua
        return x_train[feat], y_train


class NoFS(FeaturesSelection):

    def select_features(self, k: int, x_train: DataFrame, y_train: Series) -> (DataFrame, Series):
        return x_train, y_train


if __name__ == "__main__":
    dataset = HouseDataset(0)
    _, feats = dataset.get_features_with_separated_id()
    feature = NoFS()

    x_select, y_select = feature.select_features(10)
    print("x ", x_select)
    print("y ", y_select)
