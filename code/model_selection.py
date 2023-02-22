from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.svm import SVR

from feature_selection import FeaturesSelection, SelectKBestFS, NoFS
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, RidgeCV, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error

from dataset import HouseDataset
from abc import ABC, abstractmethod


class SCALER(Enum):
    # Standardizzazione dei dati in modo da avere media 0 e varianza 1
    STANDARD_SCALER = 1
    ROBUST_SCALER = 2
    NONE = 3


class MODEL(Enum):
    LINEAR_REGRESSION = 1
    LASSO_REGRESSION = 2


class ModelSelection(ABC):
    """
    L'idea di questa interfaccia è quella di selezionare il modello migliore e poi usare il modello migliore
    con i parametri imparati per fare le predizioni
    """

    def __init__(self, dataset: HouseDataset, k_folds=10):
        self.ids, self.x_train = dataset.get_features_with_separated_id()
        self.y_train = dataset.get_target()
        self.name: str = ""
        self.k_folds = k_folds
        # valori di mse e r^2 al variare del numero di feature scelte
        self.features_scores = pd.DataFrame(columns=['n_features', 'mse', 'r2'])
        self.model = None
        self.scaler = None

    @abstractmethod
    def select_model(self, fs: FeaturesSelection) -> Any:
        """
        Seleziona i parametri migliori per il modello, eseguendo anche feature selection
        :param fs:
        :return: il numero migliore di feature da usare ed eventuali iperparametri
        """
        pass

    def match_scaler(self, x_selected):
        """
        Trasforma il dataset standardizzando le feature
        La standardizzazione è un'operazione di preprocessing che consiste nel trasformare i dati in modo da avere
        una media pari a zero e una deviazione standard pari a uno. Vengono portate le feature sulla stessa scala
        e si garantisce che i valori di tutte le feature abbiano la stessa importanza per il modello.
        :param x_selected:
        :param feat_scaler_type: il tipo di scaler da scegliere o SCALER.NONE per non standardizzare nulla
        :param x_train: il dataset di training senza gli id
        :return:
        """
        match self.scaler:
            case SCALER.STANDARD_SCALER:
                # normalizza le variabili con media e deviazione standard
                scaler = StandardScaler()
                x_train_scaled = scaler.fit_transform(x_selected)
            case SCALER.ROBUST_SCALER:
                # normalizza le variabili con mediana e mediana assoluta delle deviazioni (migliore con gli outlier)
                scaler = RobustScaler()
                x_train_scaled = scaler.fit_transform(x_selected)
            case _:
                x_train_scaled = x_selected
        return x_train_scaled


class LinearRegressionMS(ModelSelection):

    def __init__(self, dataset: HouseDataset, scaler: SCALER):
        super().__init__(dataset)
        self.scaler = scaler
        self.name = "LinearRegression"
        self.model = LinearRegression

    def select_model(self, fs: FeaturesSelection) -> (DataFrame, int):
        """
        Esegue model selection per un modello LinearRegression
        :param fs: un oggetto che implementa FeaturesSelection
        :return: il DataFrame con le feature selezionate
        """
        # Cerchiamo il valore k ottimale di features da selezionare in base al valore di R^2
        best_r2 = -float('inf')
        best_features = None
        best_k = 1
        for k in range(1, self.x_train.shape[1] + 1):
            model = self.model()

            # Selezioniamo le k features migliori utilizzando la regressione univariata
            x_selected, y_select = fs.select_features(k, self.x_train, self.y_train)

            # proviamo con uno dei due scalers se usiamo la fs SelectKBest
            if isinstance(fs, SelectKBestFS):
                x_train_scaled = self.match_scaler(x_selected)
            else:
                # se no non scaliamo nulla
                x_train_scaled = self.x_train

            # fittiamo il modello scalato
            model.fit(x_train_scaled, y_select)

            # mettiamo insieme [num_features, mse, r2] e aggiungiamo al dataframe pandas
            row = [k, mean_squared_error(y_select, model.predict(x_train_scaled)),
                   model.score(x_train_scaled, y_select)]

            # prendo R^2
            r2 = row[2]

            # Questo dataframe ci sarà utile per il grafico
            self.features_scores.loc[len(self.features_scores)] = row

            # se otteniamo un r2 maggiore, allora lo salviamo e aggiorniamo le features migliori
            if r2 > best_r2:
                best_r2 = r2
                best_features = x_selected
                best_k = k

        return best_features, best_k


class LassoRegressionMS(ModelSelection):

    def __init__(self, dataset: HouseDataset, scaler):
        super().__init__(dataset)
        self.name = "LassoRegression"
        self.scaler = scaler
        self.cv_model = LassoCV
        self.model = Lasso

    def select_model(self, fs: FeaturesSelection) -> DataFrame:
        """
        Esegue la regressione lasso con k-fold cross-validation per determinare
        il miglior valore di alpha
        :param fs: oggetto che implementa FeatureSelection
        :return:
        """
        # LASSO CV
        domain = np.linspace(0, 10, 100)
        cv = 10
        scores = []
        kf = KFold(n_splits=cv)
        # considera tutti i valori di alpha in domain
        for a in domain:
            # definisce modello con Lasso
            p = Pipeline([('scaler', StandardScaler()), ('regression', Lasso(alpha=a))])
            xval_err = 0
            # per ogni coppia train-test valuta l'errore sul test set del modello istanziato sulla base del training set
            for k, (train_index, test_index) in enumerate(kf.split(X, y)):
                p.fit(X[train_index], y[train_index])
                y1 = p.predict(X[test_index])
                err = y1 - y[test_index]
                xval_err += np.dot(err, err)
            # calcola erroe medio
            score = xval_err / X.shape[0]
            scores.append([a, score])
        scores = np.array(scores)
        return y_pred


class GeneralizedLassoRegressionMS(ModelSelection):

    def __init__(self, dataset: HouseDataset, scaler):
        super().__init__(dataset)
        self.name = "GeneralizedLassoRegression"
        self.scaler = scaler
        self.model = Lasso()  # TODO: trovare generalized lasso
        self.alpha = 0.0

    def select_model(self, fs: FeaturesSelection):
        """
        Questa usa una funzione kernel giusto???
        :param alpha:
        :return:
        """
        pass


class RidgeRegressionMS(ModelSelection):

    def __init__(self, dataset: HouseDataset, scaler):
        super().__init__(dataset)
        self.scaler = scaler
        self.name = "RidgeRegression"
        self.model = Ridge()  # L2 polinomiale
        self.alpha = 0.0  # TODO: grado del polinomio?

    def select_model(self, fs: FeaturesSelection):
        """
        Esegue la regressione Ridge con k-fold cross-validation per determinare
        il miglior valore di alpha
        :return:
        """
        # tecnica di preprocessing dei dati che viene utilizzata
        # per ridimensionare i dati in modo che abbiano una
        # media zero e una deviazione standard unitaria
        x_train_scaled = self.__match_scaler()

        # crea istanza del modello di regressione Lasso
        ridge_cv = RidgeCV(cv=7)

        # addestra il modello
        ridge_cv.fit(x_train_scaled, self.y_train)

        # coefficiente di regressione: identifica le variabili che il modello ritiene più importanti per la predizione
        reg_coef = ridge_cv.coef_

        # calcolo errore quadratico medio negativo (neg_mean_squared_error)
        # utilizziamo questo perchè cross_val_score cerca di massimizzare il punteggio,
        # mentre per l'MSE vogliamo minimizzarlo.
        cross_val_score(estimator=ridge_cv, X=x_train_scaled, y=self.y_train, cv=20,
                        scoring='neg_mean_squared_error')

        # ridge_cv.alpha_: miglior valore di alpha
        return ridge_cv.alpha_, reg_coef


class ElasticNetRegressionMS(ModelSelection):

    def __init__(self, dataset: HouseDataset, scaler):
        super().__init__(dataset)
        self.name = "ElasticNetRegression"
        self.scaler = scaler
        self.model = ElasticNet()  # L1 e L2 combinate
        self.alpha = 0.0  # TODO: grado del polinomio?

    def select_model(self, fs: FeaturesSelection):
        """
        Anche questa dovrebbe avere buoni risultati
        :return:
        """
        pass


class SupportVectorRegressionMS(ModelSelection):
    def __init__(self, dataset: HouseDataset, scaler, degree):
        super().__init__(dataset)
        self.name = "SVR"
        self.scaler = scaler
        self.model = SVR(kernel="poly", degree=degree)  # SVR polinimiale
        self.alpha = 0.0  # TODO: grado del polinomio?

    def select_model(self, fs: FeaturesSelection):
        """
        Support Vector Machine per la regressione
        :return: null
        """
        svr = SVR(kernel="poly", degree=1)  # parametro di regolarizzazione per ridurre overfitting
        svr.fit(self.x_train, self.y_train)
        # uso del coefficiente R^2: valori in (0,1) più è grande meglio i dati si avvicinano al modello
        acc = svr.score(self.x_test, self.y_train)
        print("SVM R^2 accuracy = ", acc)  # TODO correggi


class NeuralNetworkMS(ModelSelection):
    def __init__(self, dataset: HouseDataset):
        super().__init__(dataset)
        self.epochs = 15
        self.optimizer = "sgd"
        self.loss = "mean_squared_error"

    def select_model(self, fs: FeaturesSelection) -> Any:
        # TODO: normalizzare il dataset: la loss è enorme!!!
        n_features = dataset.get_n_features()
        x_train = dataset.get_features_with_separated_id()[1][200:]
        print(x_train.columns)
        y_train = dataset.get_target()[200:]
        print(y_train)
        x_test = dataset.get_features_with_separated_id()[1][:200]
        y_test = dataset.get_target()[:200]

        # Normalizzo il dataset
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        x_train = (x_train - mean) / std

        mean = np.mean(y_train, axis=0)
        std = np.std(y_train, axis=0)
        y_train = (y_train - mean) / std

        mean = np.mean(x_test, axis=0)
        std = np.std(x_test, axis=0)
        x_test = (x_test - mean) / std

        mean = np.mean(y_test, axis=0)
        std = np.std(y_test, axis=0)
        y_test = (y_test - mean) / std

        model = tf.keras.Sequential([
            # uso [n_features] neuroni nel primo layer
            tf.keras.layers.Dense(units=140, input_shape=[n_features-1], activation=tf.nn.relu),
            # uso un neurone nell'ultimo layer. Anche qui uso la relu, per evitare di avere dei SalePrice negativi
            #tf.keras.layers.Dense(units=400, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=1, activation=tf.nn.relu)
        ])
        # compilo il modello con ottimizzatore e funzione loss
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        # addestro la rete neurale dal 200esimo elemento in poi
        model.fit(x_train, y_train, epochs=self.epochs)
        print("testing:")
        # i primi 200 elementi li uso come testing
        model.evaluate(x_test, y_test)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.6:
            self.model.stop_training = True


if __name__ == '__main__':
    from models import Model

    dataset = HouseDataset()
    # eseguiamo un model selection con scaling standard e 10-folds cv
    # ms = LinearRegressionMS(dataset, SCALER.STANDARD_SCALER)
    #
    # # istanziamo il modello migliore
    # best_linear_regression = Model(SelectKBestFS(), ms)
    # result = best_linear_regression.apply_model()
    #
    # # stampo il risultato della model selection
    # print(ms.features_scores)

    ms = NeuralNetworkMS(dataset)
    ms.select_model(fs=NoFS())
