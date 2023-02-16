import pandas as pd
from pandas import DataFrame, Series


class HouseDataset:

    def __init__(self, preprocessing=True):
        self.train: DataFrame = pd.read_csv('../dataset/train.csv')
        print(len(self.train.columns))
        # TODO: ci sono colonne in meno nel testing
        self.test: DataFrame = pd.read_csv('../dataset/test.csv')
        print(len(self.test.columns))
        self.selected_features: int = 0
        if preprocessing:
            self.__preprocessing()

    def get_features(self) -> DataFrame:
        """
        :return: Un DataFrame con le sole features del training set. Per ottenere il Dataframe del testing set, usa self.test
        """
        return self.train.iloc[:, :-1]

    def get_features_with_separated_id(self) -> (Series, DataFrame):
        return self.train.iloc[:, 0], self.train.iloc[:, 1:-1]

    def get_target(self) -> Series:
        """
        :return: Una Series con solo la colonna target del training set.
        """
        return self.train.iloc[:, -1]

    def get_n_features(self) -> int:
        """
        :return: restituisce il numero di features nel training set (comprese le one-hot encoded). Corrisponde al numero di colonne nel testing set
        """
        return len(self.train.columns) - 1

    def get_n_columns(self) -> int:
        """
        :return: restituisce il numero di colonne totale nel training set
        """
        return len(self.train.columns)

    def __one_hot_encoding(self, dataset: DataFrame) -> DataFrame:
        """
        Mappa le features categoriche in nuove colonne in formato one-hot
        :param dataset: self.train o self.test
        :return: il DataFrame di train o test con le stringhe trasformate in ulteriori colonne one-hot (con i nomi)
        """
        # applica il One Hot Encoding alle colonne categoriche
        encoded_df = pd.get_dummies(dataset)

        # aggiunge il prefisso al nome delle colonne
        prefix_dict = {col: f"{col}_{val}" for col in dataset.columns for val in dataset[col].unique()}

        encoded_df.add_prefix('').rename(columns=prefix_dict)

        return encoded_df

    def __fill_nan(self, dataset: DataFrame) -> DataFrame:
        """
        Riempie i valori nulli con la media per le colonne numeriche e con il valore più comune per le colonne categoriche
        :param dataset: self.train o self.test
        :return: il dataframe di train o test senza valori nulli.
        """
        # identifichiamo le colonne categoriche e numeriche
        categoriche = dataset.select_dtypes(include='object').columns
        numeriche = dataset.select_dtypes(include='number').columns

        # TODO: fare attenzione, forse la media non va bene per tutte le colonne numeriche in cui ci sono NaN
        # TODO: fare attenzione, forse non va bene usare il valore più comune per i valori nulli sulle colonne categoriche

        # sostituiamo i valori nulli con la moda per le colonne categoriche
        dataset[categoriche] = dataset[categoriche].fillna(dataset[categoriche].mode().iloc[0])

        # sostituiamo i valori nulli con la media per le colonne numeriche
        dataset[numeriche] = dataset[numeriche].fillna(dataset[numeriche].mean())

        return dataset

    def __remove_nan_value(self, dataset: DataFrame, threshold: int):
        """Elimina colonne con valori NaN quando sono tanti"""
        # eliminiamo colonne che contengono valori Nan maggiori del 20%
        thresh = int((len(dataset) * threshold) / 100) + 1
        # axis: specifichiamo di eliminare solo le colonne; thresh: numero minimo per eliminare
        dataset.dropna(axis='columns', thresh=thresh, inplace=True)
        return dataset

    def __preprocessing(self):
        """ Metodo che effettua il preprocessing sui due split"""
        self.train = self.__remove_nan_value(self.train, threshold=20)
        self.test = self.__remove_nan_value(self.test, threshold=20)

        self.train = self.__fill_nan(self.train)
        self.test = self.__fill_nan(self.test)

        self.train = self.__one_hot_encoding(self.train)
        self.test = self.__one_hot_encoding(self.test)

        # mettiamo la colonna target alla fine
        target = self.train.pop('SalePrice')
        self.train['SalePrice'] = target


if __name__ == "__main__":
    # current_directory = os.getcwd()
    # print("La cartella corrente è:", current_directory)

    dataset = HouseDataset()
    # features = dataset.get_features()
    # target = dataset.get_target()
