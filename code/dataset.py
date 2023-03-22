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

        self.selected_features: int = 0
        self.threshold = threshold
        self.__preprocessing()

    def get_features_with_id(self) -> DataFrame:
        """
        :return: Un DataFrame con le sole features del training set (contiene anche i nomi delle colonne).
        Per ottenere il Dataframe del testing set, usa self.test
        """
        return self.train.iloc[:, :-1]

    def get_features_with_separated_id(self) -> (Series, DataFrame):
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

    def __ordinal_encoding(self, dataset: DataFrame) -> DataFrame:
        # applica il One Hot Encoding alle colonne categoriche
        categoriche = dataset.select_dtypes(include='object').columns

        for cat in categoriche:
            dataset[cat], uniques = pd.factorize(dataset[cat])

        return dataset

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

        count_nan = dataset.isnull().sum(axis=0)
        count_nans_filtered = count_nan.loc[count_nan != 0]

        if len(count_nans_filtered) != 0:
            # sostituiamo i valori nulli con la moda per le colonne categoriche
            dataset[categoriche] = dataset[categoriche].fillna(dataset[categoriche].mode().iloc[0])

            # sostituiamo i valori nulli con la media per le colonne numeriche
            dataset[numeriche] = dataset[numeriche].fillna(dataset[numeriche].mean())

        return dataset

    def __remove_nan_value(self, dataset: DataFrame):
        """Elimina colonne con valori NaN quando sono tanti"""
        # eliminiamo colonne che contengono valori Nan maggiori del 20%
        thresh = int((len(dataset) * self.threshold) / 100) + 1
        # axis: specifichiamo di eliminare solo le colonne; thresh: numero minimo per eliminare
        dataset.dropna(axis='columns', thresh=thresh, inplace=True)
        return dataset

    def __preprocessing(self):
        """ Metodo che effettua il preprocessing sui due split"""
        self.train = self.__remove_nan_value(self.train)
        self.test = self.__remove_nan_value(self.test)

        self.train = self.__fill_nan(self.train)
        self.test = self.__fill_nan(self.test)

        if ORDINAL_ENCODING:
            self.train = self.__ordinal_encoding(self.train)
            self.test = self.__ordinal_encoding(self.test)
        else:
            self.train = self.__one_hot_encoding(self.train)
            self.test = self.__one_hot_encoding(self.test)

        # mettiamo la colonna target alla fine
        target = self.train.pop('SalePrice')
        self.train['SalePrice'] = target


if __name__ == "__main__":
    # current_directory = os.getcwd()
    # print("La cartella corrente è:", current_directory)

    houseDataset = HouseDataset()

    print(houseDataset.get_categorical())
    print(houseDataset.get_categorical(raw=False))

    # features = dataset.get_features()
    # target = dataset.get_target()
