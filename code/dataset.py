import os

import pandas as pd
from pandas import DataFrame, Series
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class HouseDataset:

    def __init__(self):
        self.train: DataFrame = pd.read_csv('../dataset/train.csv')
        # TODO: eseguire il preprocessing per test-set
        self.test: DataFrame = pd.read_csv('../dataset/test.csv')
        self.n_features: int = len(self.train.columns)
        self.selected_features: int = 0
        self.__preprocessing()

    def get_features(self) -> DataFrame:
        return self.train.iloc[:, :-1]

    def get_target(self) -> Series:
        return self.train.iloc[:, -1]

    def __preprocessing(self):
        """ Metodo che effettua one hot encoding su valori stringhe e effettua la sostituzione dei
        valori mancanti
        :param dataset:
        :return:
        """
        # salviamo in due liste diverse le colonne che contengono valori numerici e categorici
        numerical_columns = [col for col in self.train.iloc[:, :-1].columns if
                             self.train[col].dtype in ["int64", "float64"]]
        categorical_columns = [col for col in self.train.columns if self.train[col].dtype == "object"]

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
        ndarray_dataset = preprocessor.fit_transform(self.train).join(self.train.iloc[:, -1]) # scegliamo la colonna sui prezzi
        self.train = ndarray_dataset
        # self.train = pd.concat([pd.DataFrame(ndarray_dataset), self.train.iloc[:, -1]], axis=1)


if __name__ == "__main__":
    current_directory = os.getcwd()
    print("La cartella corrente è:", current_directory)

    dataset = HouseDataset()
    features = dataset.get_features()
    target = dataset.get_target()

    print(features.shape)
    print(target.shape)
    print(target)

    print(dataset.train)
