import pandas as pd
from pandas import DataFrame, Series
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from dataset import HouseDataset

from enum import Enum


class FS(Enum):
    SELECT_K_BEST = 1
    MUTUAL_INFORMATION = 2


class FeatureSelection:

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def select_kbest(self, top_features: int) -> (DataFrame, Series):
        x, y = self.x_train, self.y_train

        # seleziona (K) top features
        selector = SelectKBest(f_regression, k=top_features)
        x_new = selector.fit_transform(x, y)

        # ottengo un vettore di 0,1 dove 1 indica i nomi delle colonne selezionate
        mask = selector.get_support()
        # accedo ai nomi delle colonne
        selected_features = x.columns[mask]

        # crea un nuovo dataframe con le colonne selezionate
        x_new_df = pd.DataFrame(x_new, columns=selected_features)

        # converte data in tensor
        # x_new_tensor: Tensor = torch.from_numpy(x_new).float()
        # y_tensor: Tensor = torch.from_numpy(y.values).float()
        # TODO convertire in Dataframe??? Oppure lasciare tensor?
        return x_new_df, y

    def mutual_information(self, k: int) -> (DataFrame, Series):
        """
        La quantità di informazioni che ogni feature da rispetto alle altre
        :return: le features selezionate
        """
        # calcola la informazione mutua tra le features rispetto al valore target
        mi = mutual_info_regression(self.x_train, self.y_train)
        # crea il dataframe dei risultati con la mutua informazione
        dmi = pd.DataFrame(mi, index=self.x_train.columns, columns=['mi']).sort_values(by='mi', ascending=False)
        # prende dal datagrame le k feature con più informazione mutua
        feat = list(dmi.index[:k])
        # restituisco dal training solo le k feature con più informazione mutua
        return self.x_train[feat], self.y_train


if __name__ == "__main__":
    dataset = HouseDataset()
    _, feats = dataset.get_features_with_separated_id()
    feature = FeatureSelection(feats, dataset.get_target())

    x_select, y_select = feature.select_kbest(10)
    print("x ", x_select)
    print("y ", y_select)
