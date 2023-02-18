from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.svm import SVR


from feature_selection import FeaturesSelection, SelectKBestFS
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

    def select_model(self, fs: FeaturesSelection) -> DataFrame:
        """
        Esegue model selection per un modello LinearRegression
        :param fs: un oggetto che implementa FeaturesSelection
        :return: il DataFrame con le feature selezionate
        """
        # Cerchiamo il valore k ottimale di features da selezionare in base al valore di R^2
        best_r2 = -float('inf')
        best_features = None
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

        return best_features


class LassoRegressionMS(ModelSelection):

    def __init__(self, dataset: HouseDataset, scaler):
        super().__init__(dataset)
        self.name = "LassoRegression"
        self.scaler = scaler
        self.cv_model = LassoCV()
        self.model = Lasso()

    def select_model(self, fs: FeaturesSelection) -> DataFrame:
        """
        Esegue la regressione lasso con k-fold cross-validation per determinare
        il miglior valore di alpha
        :param fs: oggetto che implementa FeatureSelection
        :return:
        """
        # LASSO CV

        best_alpha = 0.0

        for k in range(1, self.x_train.shape[1] + 1):
            # Selezioniamo le k features migliori utilizzando la regressione univariata
            x_select, y_select = fs.select_features(k, self.x_train, self.y_train)

            # Lo scaling è una tecnica di preprocessing dei dati che viene utilizzata
            # per ridimensionare i dati in modo che abbiano una
            # media zero e una deviazione standard unitaria
            x_train_scaled = self.match_scaler(x_select)

            # crea istanza del modello di regressione Lasso
            lasso_cv = LassoCV(cv=self.k_folds, random_state=42)

            # addestra il modello
            lasso_cv.fit(x_train_scaled, y_select)

            # coefficiente di regressione: identifica le variabili che il modello ritiene più importanti per la predizione
            reg_coef = lasso_cv.coef_

            # calcolo errore quadratico medio negativo (neg_mean_squared_error)
            # utilizziamo questo perchè cross_val_score cerca di massimizzare il punteggio,
            # mentre per l'MSE vogliamo minimizzarlo.
            cross_val_score(estimator=lasso_cv, X=x_train_scaled, y=self.y_train, cv=20,
                            scoring='neg_mean_squared_error')

            best_alpha = lasso_cv.alpha_  # miglior valore di alpha

        # LASSO
        # Crea un'istanza del modello di regressione Lasso
        lasso = Lasso(alpha=self.cv_model.alpha_)
        lasso.fit(x_train_scaled, y_select)
        # Valutiamo le prestazioni del modello sul test set
        r2 = lasso.score(x_train_scaled, y_select)
        mse = cross_val_score(estimator=lasso, X=x_train_scaled, y=y_select, cv=self.k_folds,
                              scoring='neg_mean_squared_error')

        y_pred = lasso.predict(x_train_scaled)
        # result = Result(self.name, r2, mse, y_pred)
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


class PolynomialRegressionMS(ModelSelection):
    def __init__(self, dataset: HouseDataset, scaler, model_type: MODEL, alpha=0.0):
        super().__init__(dataset)
        self.name = "PolynomialRegression"
        self.scaler = scaler
        match model_type:
            case MODEL.LINEAR_REGRESSION:
                model = LinearRegression()
            case MODEL.LASSO_REGRESSION:
                model = Lasso(alpha=alpha)
            case _:
                print("Devi scegliere tra Lasso e LinearRegression")
                return

        self.model = model
        self.alpha = alpha

    def select_model(self, fs: FeaturesSelection):
        # ex polnomial_features()
        results = []

        # tecnica di preprocessing dei dati che viene utilizzata
        # per ridimensionare i dati in modo che abbiano una
        # media zero e una deviazione standard unitaria
        x_train_scaled = self.__match_scaler()
        model = None
        x_train_pf = None

        # TODO: aggiungere feature selection
        for degree in range(1, 3):
            polynomial_features = PolynomialFeatures(degree=degree)
            x_train_pf = polynomial_features.fit_transform(x_train_scaled)

            model.fit(x_train_pf, self.y_train)

            r2 = model.score(x_train_pf, self.y_train)
            # prestazioni tramite cross-validation
            scores = cross_val_score(estimator=model, X=x_train_pf, y=self.y_train, cv=5,
                                     scoring='neg_mean_squared_error')
            mse = mean_squared_error(model.predict(x_train_pf), self.y_train)
            mse_cv = -scores.mean()

            results.append([degree, mse, mse_cv])

        y = model.predict(x_train_pf)
        # result= Result(r2, )
        return results, y


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


if __name__ == '__main__':
    from models import Model
    dataset = HouseDataset(15.0)
    # eseguiamo un model selection con scaling standard e 10-folds cv
    ms = LinearRegressionMS(dataset, SCALER.STANDARD_SCALER)

    # istanziamo il modello migliore
    best_linear_regression = Model(SelectKBestFS(), ms)
    result = best_linear_regression.apply_model()

    # stampo il risultato della model selection
    print(ms.features_scores)
