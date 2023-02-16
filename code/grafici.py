import numpy as np
from matplotlib import cm
import matplotlib.colors as mcolors
from pandas import DataFrame

from code.dataset import HouseDataset
from code.models import Model
from matplotlib import pyplot as plt


def check_nan_value(dataset):
    # Verifichiamo se ci sono valori NaN
    count_nan = dataset.isnull().sum(axis=0)
    count_nans_filtered = count_nan.loc[count_nan != 0]
    nan_cols = dataset.columns[dataset.isna().any()]

    plt.bar(nan_cols, count_nans_filtered, color="green")
    plt.xlabel("Columns")
    plt.xticks(rotation=90)
    plt.ylabel("NaNs value")
    plt.show()


class Grafici:

    def __init__(self, model: Model):
        self.model = model

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

    def linear_regression_plot(self, selected_train: DataFrame):
        y_pred = self.model.linear_regr.predict(selected_train)

        mm = min(y_pred)
        mx = max(y_pred)

        fig = plt.figure(figsize=(16, 8))
        ax = fig.gca()
        # residuo= differenza tra valore predetto e valore addestrato del training
        plt.scatter(y_pred, (y_pred - self.model.y_train), c=self.colors[8], edgecolor='xkcd:light grey')
        plt.xlabel(r'Valori predetti ($y_i$)')
        plt.ylabel(r'Residui ($y_i-t_i$)')
        plt.hlines(y=0, xmin=(int(mm) / 10) * 10, xmax=(int(mx) / 10) * 10 + 10, color=self.colors[2], lw=2)
        plt.show()
