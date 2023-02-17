import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR

from dataset import HouseDataset
from feature_selection import FeatureSelection, FS


class Model:

    def __init__(self, dataset: HouseDataset):
        self.id, self.x_train = dataset.get_features_with_separated_id()
        self.y_train = dataset.get_target()
        self.x_test = dataset.test
        self.y_test = None
        self.linear_regr = LinearRegression()

    def linear_regression_fs(self, feat_sel_type: FS):
        """
        Da scarsi risultati perché il modello non è linearmente separabile
        :return:
        """
        results_mse = []
        feature = FeatureSelection(self.x_train, self.y_train)

        for k in range(10, len(self.x_train.columns), 10):
            # selezioniamo le prime k feature
            x_select, y_select = self.__match_feature_selection(feat_sel_type, feature, k)
            # addestriamo con la regressione linare
            self.linear_regr.fit(x_select, y_select)
            # calcoliamo il mean squared error
            mse = mean_squared_error(self.linear_regr.predict(x_select), y_select)
            results_mse.append((k, mse))
            # print('MSE: {0:} {1:.3f}'.format(k, mse))
        return results_mse

    def linear_regression(self, feat_sel_type: FS, n_features: int):
        kfold = KFold(n_splits=10)

        feature = FeatureSelection(self.x_train, self.y_train)
        x_select, y_select = self.__match_feature_selection(feat_sel_type, feature, n_features)

        # addestriamo con la regressione linare
        self.linear_regr.fit(x_select, y_select)

        scores = cross_val_score(estimator=self.linear_regr, X=x_select, y=self.y_train, cv=kfold,
                                 scoring='neg_mean_squared_error')

        y_pred = self.linear_regr.predict(x_select)
        return scores, y_pred

    def lasso_regression(self, alpha: float, cv: int):
        """
        Utilizza la regressione lasso.
        Ci aspettiamo scarsi risultati anche qui (è lineare con regolarizzazione)
        :return: i risultati dei valori MSE e la predizione sui valori del training
        """
        kfold = KFold(n_splits=cv)

        # Crea un'istanza del modello di regressione Lasso
        lasso = Lasso(alpha=alpha)
        lasso.fit(self.x_train, self.y_train)
        scores = cross_val_score(estimator=lasso, X=self.x_train, y=self.y_train, cv=kfold,
                                 scoring='neg_mean_squared_error')

        y_pred = lasso.predict(self.x_train)
        return scores, y_pred

    def lasso_cv_regression(self):
        """
        Esegue la regressione lasso con k-fold cross-validation per determinare
        il miglior valore di alpha
        :return:
        """
        # crea istanza del modello di regressione Lasso
        lasso_cv = LassoCV(cv=7, random_state=42)

        # addestra il modello
        lasso_cv.fit(self.x_train, self.y_train)

        # coefficiente di regressione: identifica le variabili che il modello ritiene più importanti per la predizione
        reg_coef = lasso_cv.coef_

        # calcolo errore quadratico medio negativo (neg_mean_squared_error)
        # utilizziamo questo perchè cross_val_score cerca di massimizzare il punteggio,
        # mentre per l'MSE vogliamo minimizzarlo.
        cross_val_score(estimator=lasso_cv, X=self.x_train, y=self.y_train, cv=20,
                        scoring='neg_mean_squared_error')

        # lasso_cv.alpha_: miglior valore di alpha
        return lasso_cv.alpha_, reg_coef

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

    def ridge_regression_cv(self):
        """
          Esegue la regressione Ridge con k-fold cross-validation per determinare
          il miglior valore di alpha
          :return:
                """
        # crea istanza del modello di regressione Lasso
        ridge_cv = RidgeCV(cv=7, random_state=42)

        # addestra il modello
        ridge_cv.fit(self.x_train, self.y_train)

        # coefficiente di regressione: identifica le variabili che il modello ritiene più importanti per la predizione
        reg_coef = ridge_cv.coef_

        # calcolo errore quadratico medio negativo (neg_mean_squared_error)
        # utilizziamo questo perchè cross_val_score cerca di massimizzare il punteggio,
        # mentre per l'MSE vogliamo minimizzarlo.
        cross_val_score(estimator=ridge_cv, X=self.x_train, y=self.y_train, cv=20,
                        scoring='neg_mean_squared_error')

        # lasso_cv.alpha_: miglior valore di alpha
        return ridge_cv.alpha_, reg_coef

    def elastic_net_regression(self):
        """
        Anche questa dovrebbe avere buoni risultati
        :return:
        """
        pass

    def svm(self):
        """
        Support Vector Machine per la regressione
        :return: null
        """
        svr = SVR(kernel="linear", C=1.0)  # parametro di regolarizzazione per ridurre overfitting
        svr.fit(self.x_train, self.y_train)
        # uso del coefficiente R^2: valori in (0,1) più è grande meglio i dati si avvicinano al modello
        acc = svr.score(self.x_test, self.y_train)
        print("SVM R^2 accuracy = ", acc)  # TODO correggi

    def __match_feature_selection(self, feat_sel_type: FS, feature: FeatureSelection, k: int) -> (DataFrame, Series):
        # selezioniamo le prime k feature
        match feat_sel_type:
            case FS.SELECT_K_BEST:
                x_select, y_select = feature.select_kbest(k)
            case FS.MUTUAL_INFORMATION:
                x_select, y_select = feature.mutual_information(k)
            case _:
                print("MUORI")
                return
        return x_select, y_select


if __name__ == "__main__":
    dataset = HouseDataset()
    models = Model(dataset)
    # print("FS.SELECT_K_BEST")
    models.linear_regression_fs(FS.SELECT_K_BEST)
    # print("FS.MUTUAL_INFORMATION")
    # models.linear_regression(FS.MUTUAL_INFORMATION)
    # print("Lasso regression")
    # models.lasso_cv_regression()
