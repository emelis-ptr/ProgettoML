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

    def correlation_matrix(self, df_train, k=10):
        #saleprice correlation matrix
        corrmat = df_train.corr()
        cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
        cm = np.corrcoef(df_train[cols].values.T)
        sns.set(font_scale=1.25)
        hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
        plt.show()


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
