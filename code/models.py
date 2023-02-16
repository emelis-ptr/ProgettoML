from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

from code.dataset import HouseDataset
from code.feature_selection import FeatureSelection


class Model:

    def __init__(self, dataset: HouseDataset):
        self.x_train = dataset.get_features()
        self.y_train = dataset.get_target()
        self.x_test = dataset.test
        self.y_test = None
        self.linear_regr = LinearRegression()

    def linear_regression(self):
        """
        Da scarsi risultati perché il modello non è linearmente separabile
        :return:
        """
        results_mse = []
        feature = FeatureSelection(self.x_train, self.y_train)
        for k in range(10, len(self.x_train.columns), 10):
            # selezioniamo le prime k feature
            x_select, y_select = feature.select_kbest(k)
            # addestriamo con la regressione linare
            self.linear_regr.fit(x_select, y_select)
            # calcoliamo il mean squared error
            mse = mean_squared_error(self.linear_regr.predict(x_select), y_select)
            results_mse.append((k, mse))
            print('MSE: {0:} {1:.3f}'.format(k, mse))

    def lasso_regression(self, alpha):
        """
        Ci aspettiamo scarsi risultati anche qui (è lineare con regolarizzazione)
        :param alpha:
        :return:
        """
        pass

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


if __name__ == "__main__":
    dataset = HouseDataset()
    models = Model(dataset)
    models.linear_regression()
