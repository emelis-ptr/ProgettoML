from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

from code.dataset import HouseDataset


class Model:

    def __init__(self, dataset: HouseDataset):
        self.x_train = dataset.get_features()
        self.y_train = dataset.get_target()
        self.x_test = dataset.test
        self.y_test = None
        self.r = LinearRegression()

    def linear_regression(self):
        self.r.fit(self.x_train, self.y_train)
        print('MSE: {0:.3f}'.format(mean_squared_error(self.r.predict(self.x_train), self.y_train)))

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
