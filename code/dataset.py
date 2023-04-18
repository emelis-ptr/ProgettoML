from typing import Tuple
import pandas as pd
from pandas import DataFrame, Series, Index

ORDINAL_ENCODING = True

class HouseDataset:

    def __init__(self, threshold=15.0):
        """
        Crea un'istanza di HouseDataset
        :param threshold: la percentuale di valori nulli sul totale di istanze dopo la quale verrà eliminata una colonna
        """
        # questo dataset è usato per l'addestramento. Va diviso in training, validation e testing set.
        self.raw_train: DataFrame = pd.read_csv('../dataset/train.csv')
        # questo dataset e' usato per fare l'upload su kaggle
        self.kaggle_test: DataFrame = pd.read_csv('../dataset/test.csv')
        # questo dataset lo usiamo per training e validation
        self.train = pd.read_csv('../dataset/train.csv')
        # questa porzione di dataset la usiamo per il testing
        self.test = pd.read_csv('../dataset/test.csv')
        

    def get_features_with_id(self) -> DataFrame:
        """
        :return: Un DataFrame con le sole features del training set (contiene anche i nomi delle colonne).
        Per ottenere il Dataframe del testing set, usa self.test
        """
        return self.train.iloc[:, :-1]

    def get_features_with_separated_id(self) -> Tuple[Series, DataFrame]:
        """
        :return: Una Series e un DataFrame con le sole features del training set (contiene anche i nomi delle colonne).
        Per ottenere il Dataframe del testing set, usa self.test
        """
        return self.train.iloc[:, 0], self.train.iloc[:, 1:-1]

    def get_target(self) -> Series:
        """
        :return: Una Series con solo la colonna target del training set (compreso il nome).
        """
        return self.train.iloc[:, -1]

    def get_n_features(self, raw=False) -> int:
        """
        :param raw: Se True, restituisce il numero di feature escluso l'id prima del preprocessing
        :return: Restituisce il numero di features nel training set per default dopo il preprocessing (comprese le one-hot encoded).
        Corrisponde al numero di colonne nel testing set - 1
        """
        return self.get_n_columns(raw) - 2

    def get_n_features_with_id(self, raw=False) -> int:
        """
        :param raw: Se True, restituisce il numero di feature compreso l'id prima del preprocessing
        :return: Restituisce il numero di features nel training set per default dopo il preprocessing (comprese le one-hot encoded e l'id).
        Corrisponde al numero di colonne nel testing set
        """
        return self.get_n_columns(raw) - 1

    def get_n_columns(self, raw=False) -> int:
        """
        :param raw: Se True, restituisce il numero feature+target compreso l'id prima del preprocessing
        :return: Restituisce il numero di colonne totale nel training set, dopo il preprocessing (per default)
        """
        return len(self.train.columns) if not raw else len(self.raw_train.columns)

    def get_categorical(self, raw=True) -> Index:
        """
        Per default, restituisce un Index coi nomi delle colonne categoriche del dataset iniziale.
        Se raw=False restituisce nessuna colonna, perché il dataset dopo il preprocessing non ha colonne categoriche.
        :param raw:
        :return:
        """
        if raw:
            return self.raw_train.select_dtypes(include='object').columns
        else:
            return self.train.select_dtypes(include='object').columns

    def get_numerical(self, raw=False, include_id=False) -> Index:
        """
        Per default, restituisce un Index coi nomi delle colonne numeriche del dataset, con il target e senza l'id, dopo il
        preprocessing del dataset.

        Se raw=False sarà uguale a get_n_features, perché il dataset trasformato ha solo colonne numeriche, altrimenti
        restituisce solo le colonne che in origine erano numeriche.
        :param include_id: Se true, include anche l'id tra le feature numeriche (default False)
        :param raw: Se true, considera il dataset iniziale (default False)
        :return:
        """
        # uso l'operatore ternario per scrivere gli if in una sola riga
        train_dataset = self.raw_train if raw else self.train
        numerical_dataset = train_dataset.select_dtypes(include='number')
        return numerical_dataset.columns if include_id else numerical_dataset.columns[1:]


if __name__ == "__main__":
    # current_directory = os.getcwd()
    # print("La cartella corrente è:", current_directory)

    houseDataset = HouseDataset()

    print(houseDataset.get_categorical())
    print(houseDataset.get_categorical(raw=False))

    # features = dataset.get_features()
    # target = dataset.get_target()
