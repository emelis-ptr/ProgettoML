import torch
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from pandas import DataFrame
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import torch.nn as nn

k_folds = 5
# Set fixed random number seed
torch.manual_seed(42)

# Parametri dell'addestramento
# n_epochs: indica il numero di epoche per cui il modello viene addestrato durante
# ogni iterazione della k-fold cross validation.
n_epochs = 10
# batch_size: indica il numero di campioni di addestramento che vengono processati dal modello
# in una singola iterazione durante l'addestramento
batch_size = 32
# learning_rate: parametro che controlla l'importanza dell'aggiornamento dei pesi in base
# al valore del gradiente della funzione di loss.
learning_rate = 0.001


class Training:

    def __init__(self):
        self.train: DataFrame = pd.read_csv('../dataset/train.csv')
        self.test: DataFrame = pd.read_csv('../dataset/test.csv')
        self.n_features: int = len(self.train.columns)
        self.selected_features: int = 0

    def prepare_dataset(self):
        # dataset.drop(index=dataset.index[0], axis=0, inplace=True)
        return pd.concat([self.train, self.test])

    def prova(self):
        print("Lenght dataset: ", len(self.train))

        # eliminiamo colonne che contengono valori Nan maggiori del 20%
        threshold = int((len(self.train) * 20) / 100) + 1

        check_nan_value(self.train)
        # axis: specifichiamo di eliminare solo le colonne; thresh: numero minimo per eliminare
        self.train.dropna(axis='columns', thresh=threshold, inplace=True)

        print("Numero di colonne prima OHE: ", len(self.train.columns))

        # # convertire l'nd.array in un tensore di PyTorch
        # tensor = torch.Tensor(ndarray_dataset)
        # # creare un oggetto TensorDataset
        # dataset = TensorDataset(tensor)
        #
        # self.selected_features = len(dataset[0])
        # print(ndarray_dataset)
        # for fold, (train_index, test_index) in enumerate(kfold_val().split(dataset)):
        #     print(f'FOLD {fold}')
        #
        #     # Divide il dataset in training set e validation set
        #     train_set = torch.utils.data.Subset(dataset, train_index)
        #     val_set = torch.utils.data.Subset(dataset, test_index)
        #
        #     # Crea i DataLoader per il training set e il validation set
        #     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        #     val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        #
        #     # Crea il modello
        #     model = MyModel(self.selected_features, 100, 80)
        #
        #     # Definisci la funzione di loss e l'ottimizzatore
        #     # funzione di perdita che calcola la media degli errori quadratici tra gli input e i target.
        #     criterion = torch.nn.MSELoss()
        #     # è un algoritmo di ottimizzazione basato sul gradiente che viene utilizzato per
        #     # aggiornare i pesi della rete neurale durante il processo di addestramento
        #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        #
        #     # Addestra il modello
        #     for epoch in range(n_epochs):
        #         for x in train_loader:
        #             optimizer.zero_grad()  # azzera i gradienti dei pesi
        #             y_pred = model(x)
        #             loss = criterion(y_pred, x[:-1])
        #             loss.backward()
        #             optimizer.step()
        #
        #         # Valuta il modello sul validation set
        #         with torch.no_grad():
        #             val_loss = 0.0
        #             for X_val, y_val in val_loader:
        #                 y_val_pred = model(X_val)
        #                 val_loss += criterion(y_val_pred, y_val)
        #             val_loss /= len(val_loader)
        #             print(f"Validation loss: {val_loss}")
        #
        #         # Seleziona le migliori features usando SelectKBest e f_regression
        #         # sostituisci 5 con il numero di features che vuoi selezionare
        #         selector = SelectKBest(f_regression, k=5)
        #         selector.fit(x, x[:-1])
        #         selected_features = selector.get_support()
        #
        #         # Riduci le features del dataset
        #         self.train.reduce_features(selected_features)
        #
        #     print(self.train)


def kfold_val():
    return KFold(n_splits=k_folds, shuffle=True)

class MyModel(nn.Module):
    """Rappresenta una rete neurale con un singolo strato nascosto
    e una funzione di attivazione ReLU per un problema di regressione."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """Vengono creati due strati lineari (nn.Linear) per la
        trasformazione lineare degli input, un oggetto ReLU per l'attivazione non-lineare
        e un altro strato lineare per la trasformazione finale in output."""
        super(MyModel, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()  # funzione di attivazione non-lineare
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Viene definita la struttura della rete neurale,
        ovvero come i tensori di input vengono trasformati in tensori di output."""
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    training = Training()
    training.prova()
