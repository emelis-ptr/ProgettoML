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
from torch.utils.data import DataLoader, ConcatDataset
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
        self.train: DataFrame = pd.read_csv('dataset/train.csv')
        self.test: DataFrame = pd.read_csv('dataset/test.csv')
        self.n_features: int = len(self.train.columns)
        self.selected_features: int = 0

    def prepare_dataset(self):
        # dataset.drop(index=dataset.index[0], axis=0, inplace=True)
        return pd.concat([self.train, self.test])

    def prova(self):
        dataset = self.train
        print("Lenght dataset: ", len(dataset))

        # notiamo che alcune colonne hanno un numero elevato di valori NaN, quindi è opportuno eliminarli
        check_nan_value(dataset)

        # eliminiamo colonne che contengono valori Nan maggiori del 20%
        threshold = int((len(dataset) * 20) / 100) + 1

        # axis: specifichiamo di eliminare solo le colonne; thresh: numero minimo per eliminare
        dataset.dropna(axis='columns', thresh=threshold, inplace=True)

        x_dataset = dataset.iloc[:, :-1]
        y_dataset = dataset.iloc[:, -1]

        print("Numero di colonne prima OHE: ", len(dataset.columns))
        dataset_encoded = one_hot_encoder(dataset)

        x_tr = dataset_encoded.fit_transform(dataset)
        y_tr = pd.DataFrame(y_dataset)  # scegliamo la colonna sui prezzi

        self.selected_features = x_tr
        print(x_tr)
        for fold, (train_index, test_index) in enumerate(kfold_val().split(x_tr)):
            print(f'FOLD {fold}')

            # Divide il dataset in training set e validation set
            train_set = torch.utils.data.Subset(x_tr, train_index)
            val_set = torch.utils.data.Subset(x_tr, test_index)

            # Crea i DataLoader per il training set e il validation set
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

            # Crea il modello
            model = MyModel(self.selected_features, 100, 80)

            # Definisci la funzione di loss e l'ottimizzatore
            # funzione di perdita che calcola la media degli errori quadratici tra gli input e i target.
            criterion = torch.nn.MSELoss()
            # è un algoritmo di ottimizzazione basato sul gradiente che viene utilizzato per
            # aggiornare i pesi della rete neurale durante il processo di addestramento
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Addestra il modello
            for epoch in range(n_epochs):
                for x, y in train_loader:
                    optimizer.zero_grad()  # azzera i gradienti dei pesi
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                    loss.backward()
                    optimizer.step()

                # Valuta il modello sul validation set
                with torch.no_grad():
                    val_loss = 0.0
                    for X_val, y_val in val_loader:
                        y_val_pred = model(X_val)
                        val_loss += criterion(y_val_pred, y_val)
                    val_loss /= len(val_loader)
                    print(f"Validation loss: {val_loss}")

                # Seleziona le migliori features usando SelectKBest e f_regression
                # sostituisci 5 con il numero di features che vuoi selezionare
                selector = SelectKBest(f_regression, k=5)
                selector.fit(x, y)
                selected_features = selector.get_support()

                # Riduci le features del dataset
                dataset.reduce_features(selected_features)

            print(dataset)


def kfold_val():
    return KFold(n_splits=k_folds, shuffle=True)


def check_nan_value(dataset):
    # Verifichiamo se ci sono valori NaN
    count_nan = dataset.isnull().sum(axis=0)

    plt.bar(dataset.columns, count_nan, color="green")
    plt.xlabel("Columns")
    plt.xticks(rotation=90)
    plt.ylabel("NaNs value")
    # plt.show()


def one_hot_encoder(dataset):
    """ Metodo che effettua one hot encoding su valori stringhe e effettua la sostituzione dei
    valori mancanti
    :param dataset:
    :return:
    """
    # salviamo in due liste diverse le colonne che contengono valori numerici e categorici
    numerical_columns = [col for col in dataset.columns if dataset[col].dtype in ["int64", "float64"]]
    categorical_columns = [col for col in dataset.columns if dataset[col].dtype == "object"]

    # attraverso la Pipeline eseguiamo una serie di azioni
    # applichiamo SimpleImputer per sostituire i valori Nan con la media
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    # applichiamo SimpleImputer per sostituire i valori Nan attraverso valori più frequenti e applichiamo
    # oneHotEncoder per trasformare i valori categorici in valori numerici ognuno dei quali viene assegnato
    # un valore [0, 1]
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    # consente di applicare una sequenza di trasformazioni solo alle colonne numeriche e una sequenza separata di
    # trasformazioni solo alle colonne categoriche
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, numerical_columns),
            ('categorical', categorical_transformer, categorical_columns)
        ],
        remainder='passthrough')
    return preprocessor


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


training = Training()
training.prova()
