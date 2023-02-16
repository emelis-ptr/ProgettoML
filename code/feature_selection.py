import torch
from sklearn.feature_selection import SelectKBest, f_regression
from torch import Tensor

from code.dataset import HouseDataset
from code.models import Model


class FeatureSelection:

    def __init__(self, model: Model):
        self.model = model

    def select_kbest(self, top_features: int):
        x, y = self.model.x_train, self.model.y_train

        # seleziona top features
        selector = SelectKBest(f_regression, k=top_features)
        x_new = selector.fit_transform(x, y)

        # converte data in tensor
        # x_new_tensor: Tensor = torch.from_numpy(x_new).float()
        # y_tensor: Tensor = torch.from_numpy(y.values).float()
        # TODO convertire in Dataframe??? Oppure lasciare tensor?
        return None, None

if __name__ == "__main__":
    dataset = HouseDataset()
    model = Model(dataset)
    feature = FeatureSelection(model)

    x_select, y_select = feature.select_kbest(10)
    print("x ", x_select)
    print("y ", y_select)
