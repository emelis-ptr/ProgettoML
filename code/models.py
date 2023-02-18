from statistics import mean

from pandas import DataFrame
from sklearn.model_selection import cross_val_score

from dataset import HouseDataset
from feature_selection import FeaturesSelection, SelectKBestFS
from model_selection import ModelSelection, LinearRegressionMS, SCALER


class Result:
    """
    Questa classe contiene i risultati in termini di R^2 e mse e y_pred in modo da poterli stampare facilmente
    """

    def __init__(self, name: str, r2: float, mse: float, y_pred: [float]):
        self.name = name
        self.r2 = r2
        self.mse = mse
        self.y_pred = y_pred

    def __str__(self):
        return f"{self.name}\nr2={self.r2}\nmse={self.mse}\ny_pred={self.y_pred}"

    def __repr__(self):
        return f"{self.name}\nr2={self.r2}\nmse={self.mse}\ny_pred={self.y_pred}"


class Model:
    def __init__(self, fs: FeaturesSelection, mod_sel: ModelSelection):
        """
        Costruisce un modello dopo aver fatto model selection
        :param k_folds: numero di fold con cui fare cross validation score
        :param fs: la feature selection usata
        :param mod_sel: la model selection usata
        """
        self.name: str = mod_sel.name + fs.name
        self.model_selection = mod_sel
        self.selected_model = mod_sel.model()  # FIXME
        self.x_train: DataFrame = mod_sel.x_train
        # qua eseguiamo la selezione del modello
        self.selected_x_train: DataFrame = mod_sel.select_model(fs)
        self.y_train: DataFrame = mod_sel.y_train

    def apply_model(self) -> Result:
        # presupponiamo che la feature selection sia già stata fatta ???
        self.selected_model.fit(self.selected_x_train, self.y_train)
        r2 = self.selected_model.score(self.selected_x_train, self.y_train)
        # esegue cross validation con k_folds per calcolare il neg_mean_squared_error.
        mses: [float] = cross_val_score(estimator=self.selected_model,
                                        X=self.selected_x_train, y=self.y_train,
                                        cv=self.model_selection.k_folds,
                                        scoring="neg_mean_squared_error")
        # calcolo la media dei mean squared errors ottenute dalle k_folds
        mse = -mean(mses)
        y_pred = self.selected_model.predict(self.selected_x_train)
        return Result(self.name, r2, mse, y_pred)


if __name__ == "__main__":
    dataset = HouseDataset(15)
    ms = LinearRegressionMS(dataset, SCALER.STANDARD_SCALER)
    models = Model(SelectKBestFS(), ms)

    # print("FS.SELECT_K_BEST")

    res = models.apply_model()

    print(res.__repr__())
    # print("FS.MUTUAL_INFORMATION")
    # models.linear_regression(FS.MUTUAL_INFORMATION)
    # print("Lasso regression")
    # models.lasso_cv_regression()

    # score = models.polynomial_features()
    # print(score)
