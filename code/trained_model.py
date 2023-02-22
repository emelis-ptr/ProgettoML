from pandas import DataFrame, Series


class TrainedModel:
    """
    Questa classe rappresenta un modello addestrato dopo feature selection e model selection
    """

    def __init__(self, trained_model, name: str, r2: float, mse: float):
        self.name = name
        self.trained_model = trained_model
        self.r2 = r2
        self.mse = mse

    def __str__(self):
        return f"{self.name}\nR^2={self.r2}\nMean Square Error={self.mse}"

    def __repr__(self):
        return f"{self.name}\nr2={self.r2}\nmse={self.mse}"

    def predict(self, selected_features: DataFrame) -> Series:
        return self.trained_model.predict(selected_features)

    # TODO: prova non so come farla funzionare
    def predict_one(self, one_feature_vec) -> float:
        return self.trained_model.predict(one_feature_vec)
