import torch
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

k_folds = 5
# Set fixed random number seed
torch.manual_seed(42)


class Training:

    def __init__(self):
        self.train = pd.read_csv('../dataset/train.csv')
        self.test = pd.read_csv('../dataset/test.csv')
        self.submission = pd.read_csv('../dataset/sample_submission.csv')

    def prepare_dataset(self):
        # dataset.drop(index=dataset.index[0], axis=0, inplace=True)
        return pd.concat([self.train, self.test])

    def prova(self):
        dataset = self.train
        print("Lenght dataset: ", len(dataset))

        # notiamo che alcune colonne hanno un numero elevato di valori NaN, quindi è opportuno eliminarli
        check_Nan_value(dataset)

        # eliminiamo nel caso i valori nulli siano maggiori di un terzo del dataset; ma bisogna rivalutarlo bene
        threshold = int((len(dataset) * 20) / 100) + 1
        # axis:specifichiamo di eliminare solo le colonne; thresh:numero minimo per eliminare
        dataset.dropna(axis='columns', thresh=threshold, inplace=True)

        x_dataset = dataset.iloc[:, :-1]
        y_dataset = dataset.iloc[:, -1]

        print("Numero di colonne prima OHE: ", len(x_dataset.columns))
        dataset_encoded = one_hot_encoder(x_dataset)

        x_tr = dataset_encoded.fit_transform(x_dataset)
        y_tr = pd.DataFrame(y_dataset)  # scegliamo la colonna sui prezzi

        for fold, (train_ids, test_ids) in enumerate(kfold_val().split(dataset)):
            print(f'FOLD {fold}')


def kfold_val():
    return KFold(n_splits=k_folds, shuffle=True)


def check_Nan_value(dataset):
    # Verifichiamo se ci sono valori NaN
    count_nan = dataset.isnull().sum(axis=0)

    plt.bar(dataset.columns, count_nan, color="green")
    plt.xlabel("Columns")
    plt.ylabel("NaNs value")
    # plt.show()


def one_hot_encoder(dataset):
    """ Metodo che effettua one hot encoder su valori stringhe e effettua la sostituzione dei
    valori mancanti
    :param dataset:
    :return:
    """
    # salviamo in due liste diverse le colonne che contengono valori numerici e categorici
    numerical_columns = [col for col in dataset.columns if dataset[col].dtype in ["int64", "float64"]]
    categorical_columns = [col for col in dataset.columns if dataset[col].dtype == "object"]

    # attraverso la Pipeline eseguiamo una serie di azioni
    # applichiamo SimpleImputer per sostituire i valori Nan attraverso la media e applichiamo feature scaling
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    # applichiamo SimpleImputer per sostituire i valori Nan attraverso valori più frequenti e applichiamo
    # oneHotEncoder per trasformare i valori categorici in valori numerici ognuno dei quali viene assegnato un valore
    # [0.1] in base al valore categorico
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    # consente di applicare una sequenza di trasformazioni solo alle colonne numeriche e una sequenza separata di
    # trasformazioni solo alle colonne categoriali
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, numerical_columns),
            ('categorical', categorical_transformer, categorical_columns)],
        remainder='passthrough')
    return preprocessor


training = Training()
training.prova()
