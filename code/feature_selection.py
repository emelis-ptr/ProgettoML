import torch
from sklearn.feature_selection import SelectKBest, f_regression
from torch import Tensor

from dataset import HouseDataset


class FeatureSelection:

    def __init__(self, dataset: HouseDataset):
        self.dataset = dataset

    def select_kbest(self, top_features: int) -> (Tensor, Tensor):
        X, y = self.dataset.get_features(), self.dataset.get_target()

        # seleziona top features
        selector = SelectKBest(f_regression, k=top_features)
        X_new = selector.fit_transform(X, y)

        # converte data in tensor
        X_new = torch.from_numpy(X_new).float()
        y = torch.from_numpy(y.values).float()

        return X_new, y


if __name__ == "__main__":
    dataset = HouseDataset()
    feature = FeatureSelection(dataset)

    x, y = feature.select_kbest(10)
    print("x ", x)
    print("y ", y[20])
