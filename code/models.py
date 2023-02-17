from enum import Enum

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler
from sklearn.svm import SVR

from dataset import HouseDataset
from feature_selection import FeatureSelection, FS


class SCALER(Enum):
    # Standardizzazione dei dati in modo da avere media 0 e varianza 1
    STANDARD_SCALER = 1
    ROBUST_SCALER = 2
    NONE = 3


class MODEL(Enum):
    LINEAR_REGRESSION = 1
    LASSO_REGRESSION = 2


class Result:
    def __init__(self, r2: float, mse: [], y_pred: []):
        self.r2 = r2
        self.mse = mse
        self.y_pred = y_pred

    def __str__(self):
        return f"r2={self.r2}, mse={self.mse}, y_pred={self.y_pred}"

    def __repr__(self):
        return f"r2={self.r2}, mse={self.mse}, y_pred={self.y_pred}"


class Model:

    def __init__(self, dataset: HouseDataset, model: MODEL):
        self.id, self.x_train = dataset.get_features_with_separated_id()
        self.y_train = dataset.get_target()
        self.x_test = dataset.test
        self.y_test = None
        self.linear_regr = LinearRegression()
        self.fs = FS.NONE
        self.scaler = SCALER.NONE
        self.model = model
        self.n_features = 10
        self.k_fold = 10

    def setup_feature_selection(self, feat_sel_type: FS, scaler_type: SCALER, n_features: int):
        self.fs = feat_sel_type
        self.scaler = scaler_type
        self.n_features = n_features

    def set_model(self, model_type: MODEL):
        self.model = model_type

    def linear_regression(self):
        results_mse = []
        kfold = KFold(n_splits=self.k_fold)

        feature = FeatureSelection(self.x_train, self.y_train)
        # Selezioniamo le k features migliori utilizzando la regressione univariata
        selector, x_select, y_select = self.__match_feature_selection(self.fs, feature, self.n_features)

        if self.fs == FS.SELECT_K_BEST:
            x_train_scaled = self.__match_scaler(self.scaler, x_select)
        else:
            x_train_scaled = self.x_train

        # Addestriamo un modello di regressione lineare sulle k features selezionate
        model = LinearRegression()
        model.fit(x_train_scaled, y_select)

        # Valutiamo le prestazioni del modello sul test set
        r2 = model.score(x_train_scaled, y_select)

        # TODO: da mettere fuori
        # Cerchiamo il valore ottimale di k in base al valore di R^2
        best_score = -float('inf')
        best_k = 0
        for k in range(1, self.x_train.shape[1] + 1):
            feature = FeatureSelection(self.x_train, self.y_train)

            # Selezioniamo le k features migliori utilizzando la regressione univariata
            selector, x_select, y_select = self.__match_feature_selection(self.fs, feature, k)

            if self.fs == FS.SELECT_K_BEST:
                x_train_scaled = self.__match_scaler(self.scaler, x_select)
            else:
                x_train_scaled = self.x_train

            model.fit(x_train_scaled, y_select)
            # calcola R^2
            r2 = model.score(x_train_scaled, y_select)
            # calcola mse
            mse = mean_squared_error(y_select, model.predict(x_train_scaled))
            results_mse.append((k, mse))

            if r2 > best_score:
                best_score = r2
                best_k = k

        results_mse = np.array(results_mse)
        min_index = np.argmin(results_mse[:, 1])  # minor mse

        best_features = int(results_mse[min_index, 0])
        # print("Il miglior valore di k è: {}".format(best_k))
        # print('Minor mse: {0:5}. MSE={1:.3f}'.format(best_features, results_mse[min_index, 1]))

        mse = cross_val_score(estimator=model, X=x_train_scaled, y=y_select, cv=kfold,
                              scoring='neg_mean_squared_error')

        y_pred = model.predict(x_train_scaled)
        results = Result(r2, mse, y_pred)
        return results

    def lasso_regression(self, feat_scaler_type: SCALER):
        """
        Esegue la regressione lasso con k-fold cross-validation per determinare
        il miglior valore di alpha
        :return:
        """
        # LASSO CV
        feature = FeatureSelection(self.x_train, self.y_train)
        # Selezioniamo le k features migliori utilizzando la regressione univariata
        selector, x_select, y_select = self.__match_feature_selection(self.fs, feature, self.n_features)

        # tecnica di preprocessing dei dati che viene utilizzata
        # per ridimensionare i dati in modo che abbiano una
        # media zero e una deviazione standard unitaria
        x_train_scaled = self.__match_scaler(feat_scaler_type, x_select)

        # crea istanza del modello di regressione Lasso
        lasso_cv = LassoCV(cv=self.k_fold, random_state=42)

        # addestra il modello
        lasso_cv.fit(x_train_scaled, y_select)

        # coefficiente di regressione: identifica le variabili che il modello ritiene più importanti per la predizione
        reg_coef = lasso_cv.coef_

        # calcolo errore quadratico medio negativo (neg_mean_squared_error)
        # utilizziamo questo perchè cross_val_score cerca di massimizzare il punteggio,
        # mentre per l'MSE vogliamo minimizzarlo.
        cross_val_score(estimator=lasso_cv, X=x_train_scaled, y=self.y_train, cv=20,
                        scoring='neg_mean_squared_error')

        # lasso_cv.alpha_: miglior valore di alpha
        kfold = KFold(n_splits=self.k_fold)

        # LASSO
        # Crea un'istanza del modello di regressione Lasso
        lasso = Lasso(alpha=lasso_cv.alpha_)
        lasso.fit(x_train_scaled, y_select)
        # Valutiamo le prestazioni del modello sul test set
        r2 = lasso.score(x_train_scaled, y_select)
        mse = cross_val_score(estimator=lasso, X=x_train_scaled, y=y_select, cv=kfold,
                              scoring='neg_mean_squared_error')

        y_pred = lasso.predict(x_train_scaled)
        result = Result(r2, mse, y_pred)
        return result

    def generalized_lasso_regression(self, alpha):
        """
        Questa usa una funzione kernel giusto???
        :param alpha:
        :return:
        """
        pass

    def ridge_regression(self):
        """
        Questa dovrebbe funzionare meglio con i modelli non lineari
        :return:
        """
        pass

    def ridge_cv_regression_(self, feat_scaler_type: SCALER):
        """
          Esegue la regressione Ridge con k-fold cross-validation per determinare
          il miglior valore di alpha
          :return:
                """
        # tecnica di preprocessing dei dati che viene utilizzata
        # per ridimensionare i dati in modo che abbiano una
        # media zero e una deviazione standard unitaria
        x_train_scaled = self.__match_scaler(feat_scaler_type, self.x_train)

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

    def elastic_net_regression(self):
        """
        Anche questa dovrebbe avere buoni risultati
        :return:
        """
        pass

    def polynomial_features(self, model_type: MODEL, scaler_type: SCALER, alpha: float):
        results = []

        # tecnica di preprocessing dei dati che viene utilizzata
        # per ridimensionare i dati in modo che abbiano una
        # media zero e una deviazione standard unitaria
        x_train_scaled = self.__match_scaler(scaler_type, self.x_train)
        model = None
        x_train_pf = None

        # TODO: aggiungere feature selection
        for degree in range(1, 3):
            polynomial_features = PolynomialFeatures(degree=degree)
            x_train_pf = polynomial_features.fit_transform(x_train_scaled)

            match model_type:
                case MODEL.LINEAR_REGRESSION:
                    model = LinearRegression()
                    model.fit(x_train_pf, self.y_train)
                case MODEL.LASSO_REGRESSION:
                    model = Lasso(alpha=alpha)
                    model.fit(x_train_pf, self.y_train)
                case _:
                    print("MUORI")
                    return

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

    def svm(self):
        """
        Support Vector Machine per la regressione
        :return: null
        """
        svr = SVR(kernel="poly", degree=1)  # parametro di regolarizzazione per ridurre overfitting
        svr.fit(self.x_train, self.y_train)
        # uso del coefficiente R^2: valori in (0,1) più è grande meglio i dati si avvicinano al modello
        acc = svr.score(self.x_test, self.y_train)
        print("SVM R^2 accuracy = ", acc)  # TODO correggi

    def __match_feature_selection(self, feat_sel_type: FS, feature: FeatureSelection, k: int) -> (DataFrame, Series):
        # selezioniamo le prime k feature
        match feat_sel_type:
            case FS.SELECT_K_BEST:
                selector, x_select, y_select = feature.select_kbest(k)
            case FS.MUTUAL_INFORMATION:
                x_select, y_select = feature.mutual_information(k)
            case FS.NONE:
                print("MUORI")
                return
        return selector, x_select, y_select

    def __match_scaler(self, feat_scaler_type: SCALER, x_train):
        match feat_scaler_type:
            case SCALER.STANDARD_SCALER:
                scaler = StandardScaler()
                x_train_scaled = scaler.fit_transform(x_train)
            case SCALER.ROBUST_SCALER:
                scaler = RobustScaler()
                x_train_scaled = scaler.fit_transform(x_train)
            case SCALER.NONE:
                x_train_scaled = x_train
        return x_train_scaled


if __name__ == "__main__":
    dataset = HouseDataset(15, preprocessing=True)
    models = Model(dataset, MODEL.LINEAR_REGRESSION)
    # print("FS.SELECT_K_BEST")
    models.setup_feature_selection(FS.SELECT_K_BEST, SCALER.STANDARD_SCALER, 20)
    res = models.linear_regression()

    print(res.__repr__())
    # print("FS.MUTUAL_INFORMATION")
    # models.linear_regression(FS.MUTUAL_INFORMATION)
    # print("Lasso regression")
    # models.lasso_cv_regression()

    # score = models.polynomial_features()
    # print(score)
