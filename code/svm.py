from sklearn.svm import SVR


def svm(self):
    """
    Support Vector Machine per la regressione
    :return: null
    """

    self.preprocessing()
    svr = SVR(kernel="linear", C=1.0)  # parametro di regolarizzazione per ridurre overfitting
    svr.fit(self.X_train, self.y_train)
    acc = svr.score(self.X_test,
                    self.y_test)  # uso del coefficiente R^2: valori in (0,1) più è grande meglio i dati si avvicinano al modello
    print("SVM R^2 accuracy = ", acc) # TODO correggi