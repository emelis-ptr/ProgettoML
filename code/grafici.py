import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame

from models import Model
import seaborn as sns


class Grafici:

    def __init__(self):
        self.plt = plt
        self.plt.style.use('fivethirtyeight')
        self.plt.rcParams['font.family'] = 'sans-serif'
        self.plt.rcParams['font.serif'] = 'Ubuntu'
        self.plt.rcParams['font.monospace'] = 'Ubuntu Mono'
        self.plt.rcParams['font.size'] = 10
        self.plt.rcParams['axes.labelsize'] = 10
        self.plt.rcParams['axes.labelweight'] = 'bold'
        self.plt.rcParams['axes.titlesize'] = 10
        self.plt.rcParams['xtick.labelsize'] = 12
        self.plt.rcParams['ytick.labelsize'] = 12
        self.plt.rcParams['legend.fontsize'] = 10
        self.plt.rcParams['figure.titlesize'] = 12
        self.plt.rcParams['image.cmap'] = 'jet'
        self.plt.rcParams['image.interpolation'] = 'none'
        self.plt.rcParams['figure.figsize'] = (16, 8)
        self.plt.rcParams['lines.linewidth'] = 2
        self.plt.rcParams['lines.markersize'] = 8

        self.colors = ['xkcd:pale orange', 'xkcd:sea blue', 'xkcd:pale red', 'xkcd:sage green', 'xkcd:terra cotta',
                       'xkcd:dull purple', 'xkcd:teal', 'xkcd:goldenrod', 'xkcd:cadet blue',
                       'xkcd:scarlet']
        self.cmap_big = cm.get_cmap('Spectral', 512)
        self.cmap = mcolors.ListedColormap(self.cmap_big(np.linspace(0.7, 0.95, 256)))

        self.bbox_props = dict(boxstyle="round,pad=0.3", fc=self.colors[0], alpha=.5)

    def linear_regression_plot(self, y_train, y_pred):
        mm = min(y_pred)
        mx = max(y_pred)

        fig = self.plt.figure(figsize=(16, 8))
        ax = fig.gca()  # TODO: A che serve questa ax?
        # residuo= differenza tra valore predetto e valore addestrato del training
        self.plt.scatter(y_pred, (y_pred - y_train), c=self.colors[8], edgecolor='xkcd:light grey')
        self.plt.xlabel(r'Valori predetti ($y_i$)')
        self.plt.ylabel(r'Residui ($y_i-t_i$)')
        self.plt.hlines(y=0, xmin=(int(mm) / 10) * 10, xmax=(int(mx) / 10) * 10 + 10, color=self.colors[2], lw=2)
        self.plt.show()

    def mse_linear_regression_plot(self, scores: DataFrame):
        """
        Traccia il grafico dell'errore quadratico medio al variare del numero di features utilizzate
        :param scores: ['n_featuers', 'mse', 'r2']
        :return:
        """
        # crea due subplot con l'asse x condiviso
        fig, axs = self.plt.subplots(2, sharex='col')

        axs[0].plot(scores['n_features'], scores['mse'], linewidth=1, color=self.colors[0],
                    label='mse')  # primo subplot
        axs[1].plot(scores['n_features'], scores['r2'], linewidth=1, color=self.colors[1],
                    label='r2')  # secondo subplot
        self.plt.xlabel('Numero di feature')

        # axs[0].ylabel('MSE')
        axs[0].set_ylabel('MSE')
        axs[1].set_ylabel('R2')

        self.plt.title('MSE e R2 al variare di k-feature nella regressione lineare')
        self.plt.show()

    def mse_lasso_regression_plot(self, scores: []):
        fig = plt.figure(figsize=(16, 8))
        fig.gca()
        plt.plot(scores[:, 0], scores[:, 1])
        plt.xlabel(r'$\alpha$')
        plt.ylabel('MSE')
        plt.title(r'MSE al variare di $\alpha$ in Lasso')
        plt.show()

    def lasso_regression_plot(self, y_train, y_pred):
        mm = min(y_pred)
        mx = max(y_pred)

        fig = plt.figure(figsize=(16, 8))
        fig.gca()
        plt.scatter(y_pred, (y_pred - y_train), c=self.colors[8], edgecolor='xkcd:light grey')
        plt.xlabel(r'Valori predetti ($y_i$)')
        plt.ylabel(r'Residui ($y_i-t_i$)')
        plt.hlines(y=0, xmin=(int(mm) / 10) * 10, xmax=(int(mx) / 10) * 10 + 10, color=self.colors[2], lw=2)
        plt.tight_layout()
        plt.show()

    def mse_polynomial_features_plot(self, results):
        top = 15
        fig = plt.figure(figsize=(16, 8))
        fig.gca()
        plt.plot([r[0] for r in results[:top]], [r[1] for r in results[:top]], color=self.colors[8], label=r'Train')
        plt.plot([r[0] for r in results[:top]], [r[2] for r in results[:top]], color=self.colors[2], label=r'Test')
        plt.legend()

    def polynomial_features_plot(self, y_train, y_pred):
        mm = min(y_pred)
        mx = max(y_pred)

        fig = plt.figure(figsize=(16, 8))
        fig.gca()
        plt.scatter(y_pred, (y_pred - y_train), c=self.colors[8], edgecolor='white', label='Train')
        plt.xlabel(r'Valori predetti ($y_i$)')
        plt.ylabel(r'Residui ($y_i-t_i$)')
        plt.hlines(y=0, xmin=(int(mm) / 10) * 10, xmax=(int(mx) / 10) * 10 + 10, color=self.colors[2], lw=2)
        plt.tight_layout()
        plt.show()

    def correlation_matrix_plot(self, df_train: DataFrame, k=10):
        # saleprice correlation matrix
        corrmat = df_train.corr()
        cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
        cm = np.corrcoef(df_train[cols].values.T)
        sns.set(font_scale=1.25)
        sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                    yticklabels=cols.values, xticklabels=cols.values)
        plt.show()

    def correlation_matrix_filtered_plot(self, df_train: DataFrame, threshold=0.6):
        # threshold: valore minimo di correlazione da considerare
        val_max = 0.99
        # si selezionano le colonne
        labels = df_train.columns
        corr, ind_x, ind_y = correlazione_matrice(df_train, threshold, val_max)
        # si aggiornano le colonne selezionando solamente quelle filtrate
        map_labels_x = [item for i, item in enumerate(labels) if i not in ind_x]
        map_labels_y = [item for i, item in enumerate(labels) if i not in ind_y]
        # plot heatmap
        heatmap = sns.heatmap(corr, annot=True, fmt='.2f',
                              xticklabels=map_labels_x,
                              yticklabels=map_labels_y,
                              vmin=threshold,
                              vmax=val_max,
                              linewidths=1.0,
                              linecolor="grey")

        heatmap.set_title("Reduced heatmap")
        plt.show()


def correlazione_matrice(df_train: DataFrame, threshold: float, val_max: float):
    # determina la correlazione
    corr = df_train.corr().to_numpy()
    # seleziona gli elementi che sono compresi tra il threshold e un valore massimo
    ind_x, = np.where(np.all(np.logical_or(corr < threshold, corr > val_max), axis=0))
    corr = np.delete(corr, ind_x, 1)  # si eliminano

    # stessa cosa per le righe
    ind_y, = np.where(np.all(np.logical_or(corr < threshold, corr > val_max), axis=1))
    corr = np.delete(corr, ind_y, 0)
    # si aggiornano le colonne selezionando solamente quelle filtrate

    return corr, ind_x, ind_y


def correlazione_dataframe(df_train, threshold, val_max):
    # si selezionano le colonne
    labels = df_train.columns

    corr, ind_x, ind_y = correlazione_matrice(df_train, threshold, val_max)
    map_labels_x = [item for i, item in enumerate(labels) if i not in ind_x]
    map_labels_y = [item for i, item in enumerate(labels) if i not in ind_y]

    return pd.DataFrame(corr, columns=map_labels_x, index=map_labels_y)


def confronto(df_train, threshold, val_max):
    corr = correlazione_dataframe(df_train, threshold, val_max)

    for r in corr:
        print(r)


class Correlazione:
    def __init__(self, feature1, feature2, correlazione, correlazione_f1_target, correlazione_f2_target):
        self.feature1 = feature1
        self.feature2 = feature2
        self.correlazione = correlazione
        self.correlazione_f1_target = correlazione_f1_target
        self.correlazione_f2_target = correlazione_f2_target

    def __str__(self):
        return f"({self.feature1, self.feature2, self.correlazione, self.correlazione_f1_target, self.correlazione_f2_target})"

    def __repr__(self):
        return f"({self.feature1, self.feature2, self.correlazione, self.correlazione_f1_target, self.correlazione_f2_target})"


if __name__ == '__main__':
    from model_selection import LinearRegressionMS, SCALER
    from dataset import HouseDataset
    from feature_selection import *

    # eseguiamo un model selection con scaling standard e 10-folds cv
    h = HouseDataset()
    ms = LinearRegressionMS(h, SCALER.STANDARD_SCALER)

    # istanziamo il modello migliore
    best_linear_regression = Model(SelectKBestFS(), ms)
    result = best_linear_regression.apply_model()

    # TODO: qua ho bisogno dell'array di mse ottenuto dalla feature selection, non dalla cross validation

    # traccio il grafico del mse al variare delle features
    Grafici().mse_linear_regression_plot(ms.features_scores)
