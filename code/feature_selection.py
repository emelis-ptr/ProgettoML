import pandas as pd
from pandas import DataFrame, Series
from sklearn.feature_selection import SelectKBest, f_regression

from code.dataset import HouseDataset


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


if __name__ == "__main__":
    dataset = HouseDataset()
    feature = FeatureSelection(dataset.get_features(), dataset.get_target())

    x_select, y_select = feature.select_kbest(10)
    print("x ", x_select)
    print("y ", y_select)
