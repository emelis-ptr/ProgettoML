import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
import seaborn as sn
from pylab import subplot, imshow
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay

import torch  # pip3 install torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets  # pip3 install torchvision
# from torchvision.transforms import ToTensor
#
# from torchinfo import summary  # pip3 install torchinfo
#
# import pytorch_lightning as pl  # pip3 install pytorch-lightning
from typing import *
cmap_big = colormaps.get_cmap('Spectral')
cmap = mcolors.ListedColormap(
    cmap_big(np.linspace(0.35, 0.95, 256)))  # np.linspace crea un vettore di numeri equidistanti tra 0.35 e 0.95
cmap

plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['image.cmap'] = 'jet'
plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['lines.linewidth'] = 2

colors = ['#5D80AE', '#FC5758', '#0DD77D', '#FFC507', '#F64044',
          '#810f7c', '#137e6d', '#be0119', '#3b638c', '#af6f09']

class History:
    """Accumulates values in a dictionary of sequences."""

    def __init__(self, keys: List[str]):
        self.data: Dict[str, List[float]] = {}
        self.keys: List[str] = keys
        for k in self.keys:
            self.data[k] = []

    def add(self, *args: Any):
        for k, a in zip(self.keys, args):
            self.data[k].append(a)

    def sums(self) -> Dict[str, float]:
        """
        Ho un dizionario di chiavi e liste di float. Questo metodo produce
        un nuovo dizionario con la stessa chiave e gli elementi della sequenza sommati
        """
        return {k: sum(self.data[k]) for k in self.keys}

    def merge(self, d: dict[str, List[float]]):
        """
        Unisce il dizionario di questo oggetto con il dizionario passato in input
        :param d: un dizionario con chiave string e lista di valori float
        """
        for k in self.keys:
            self.data[k].extend(d[k])

    def __getitem__(self, k: str) -> List[float]:
        """
        Restituisce la lista relativa alla chiave k
        :param k:
        :return:
        """
        return self.data[k]


def displayData(X: np.ndarray, t: np.ndarray, rows: int = 10, cols: int = 10, size: float = 8,
                class_value: bool = False):
    """
    Mostra le immagini del dataset e il valore target subito sopra ciascuna di esse, in forma di tabella.
    :param X: Le feature, ovvero le immagini che rappresentano le cifre del dataset MNIST di training
    :param t: Il target del dataset di training, ovvero un intero tra 0,1,...,9
    :param rows: il numero di righe della tabella
    :param cols: il numero di colonne della tabella
    :param size: la dimensione della tabella
    :param class_value: se true, mostra anche i valori target al di sopra di ogni immagini (come se fosse il titolo dell'immagine)
    :return:
    """
    X = X.numpy()
    t = t.numpy()
    # se ci sono più elementi di quelli che entrano nella tabella, prende gli indici di alcuni elementi casuali
    if len(X) > rows * cols:
        # permutation(len(X)) mischia il dataset randomicamente
        img_ind = np.random.permutation(len(X))[0:rows * cols]  # poi prendo le prime rows*cols immagini
    else:
        img_ind = range(rows * cols)
    fig = plt.figure(figsize=(size, size))
    fig.patch.set_facecolor('white')
    ax = fig.gca()

    # non voglio un'immagine troppo grande, quindi prendo min(10,cols) e min(10, rows)
    num_cols = min(10, cols)
    num_rows = min(10, rows)
    num_cells = num_cols * num_rows

    for i in range(num_cells):
        # creo un subplot per ogni immagine
        plt.subplot(rows, cols, i + 1)
        # in ogni subplot voglio visualizzare l'immagine, prendendo i valori dei pixel in scala di grigi
        plt.imshow([255 - x for x in X[img_ind[i]]], cmap='gray', interpolation='gaussian')
        # opzionalmente, mostro anche il valore vero del target
        if class_value:
            plt.title("{}".format(t[img_ind[i]]), fontsize=10, color=colors[4])

        # elimina i tick su asse x e y
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.axis('off')
    plt.subplots_adjust(top=1)  # sposta il subplot verso il basso
    plt.show()


def displayOneData(X: np.ndarray, t: np.ndarray, index):
    displayData(X[index:index + 1], t[index:index + 1], rows=1, cols=1, size=1, class_value=False)

# non usato
def save_model_coeffs(m, filename):
    torch.save(m.state_dict(), filename)
    print("Saved model coefficients to disk")

# non usato
def load_model_coeffs(m, filename):
    m.load_state_dict(torch.load(filename))
    print("Model coefficients loaded from disk")

# non usato
def accuracy(preds, targets):  #@save
    """Compute the number of correct predictions."""
    # deal with the case when an array of probabilities is predict, by deriving the highest-probability class
    if len(preds.shape) > 1 and preds.shape[1] > 1:
        preds = preds.argmax(axis=1)
    cmp = preds.type(targets.dtype) == targets
    return float(cmp.type(targets.dtype).sum())

# non usato
def evaluate_accuracy(net, data_iter):
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        # Set the model to eval mode
        net.eval()
    h_test = History(['correct_predictions', 'predictions'])  # No. of correct predictions, no. of predictions
    with torch.no_grad():  # Gradients must not be computed
        count = 0
        for X, y in data_iter:
            count += 1
            if count % 10 == 0:
                print('x', end='')
            #X = X.flatten(start_dim=1, end_dim=-1)
            h_test.add(accuracy(net(X), y), len(y))
        s = h_test.sums()
    print(' ')
    return s['correct_predictions'] / s['predictions']

def train_epoch(model, train_iter, y_train, loss_func, optimizer):
    if isinstance(model, torch.nn.Module):
        model.train()  # Set the model to training mode
    h_epoch = History(
        ['loss', 'correct_predictions', 'n_examples'])  # Training loss, no. of correct predictions, no. of examples

    #X=X.flatten(start_dim=1, end_dim=-1)
    # Compute predictions
    y_hat = model(train_iter)
    # Compute loss
    loss = loss_func(y_hat, y_train)
    optimizer.zero_grad()
    # Compute gradients
    loss.backward()
    # Update parameters
    optimizer.step()
    h_epoch.add(float(loss), accuracy(y_hat, y_train), len(y_train))
    # Return training loss and training accuracy
    s = h_epoch.sums()
    return s['loss'] / s['n_examples'], s['correct_predictions'] / s['n_examples'], h_epoch

def train(net, dataset, loss_func, num_epochs, updater, report=False):
    h_batch = History(['loss', 'correct_predictions', 'n_examples'])
    h_train = History(['training_loss', 'training_accuracy',
                       'test_accuracy'])  # Avg. training loss, avg. training accuracy, test accuracy
    for epoch in range(num_epochs):
        print(f'Epoch #{epoch + 1}')
        # train model for one epoch
        train_loss, train_acc, h_epoch = train_epoch(net, dataset.X_train_normalized, dataset.y_train, loss_func, updater)
        # evaluate accuracy on test set
        test_acc = evaluate_accuracy(net, dataset.X_test_normalized)
        if report:
            print(f' Loss {train_loss:3.4f}, Training set accuracy {train_acc:1.4f}, Test set accuracy {test_acc:1.4f}')
        else:
            print('\n')
        h_train.add(train_loss, train_acc, test_acc)
        h_batch.merge(h_epoch.data)
    return h_train, h_batch

def predict(net, loaders):
    preds_train = []
    y_train = []
    preds_test = []
    y_test = []
    if isinstance(net, torch.nn.Module):
        net.eval()
    with torch.no_grad():
        for X, y in loaders['train']:
            #X = X.flatten(start_dim=1, end_dim=-1)
            preds = (torch.max(net(X), 1)[1]).numpy()
            preds_train.extend(preds)
            y_train.extend(y.numpy())
        for X, y in loaders['test']:
            #X = X.flatten(start_dim=1, end_dim=-1)
            preds = (torch.max(net(X), 1)[1]).numpy()
            preds_test.extend(preds)
            y_test.extend(y.numpy())
    return preds_train, y_train, preds_test, y_test

def plot_metrics(title, h, hb, num_epochs):
    losses = []
    accs = []
    for l, c, n in zip(hb.data['loss'], hb.data['correct_predictions'], hb.data['n_examples']):
        losses.append(l / n)
        accs.append(c / n)
    n = len(accs)
    step = int(n / num_epochs)
    xs = range(step - 1, n + step - 1, step)

    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    ax1.plot(range(len(losses)), losses, color=colors[0], lw=.5, label='Loss by batch', zorder=2)
    ax1.scatter(xs, h.data['training_loss'], color=colors[1], edgecolors='black', label='Loss by epoch', zorder=2)
    for x in xs:
        ax1.axvline(x, lw=1, color='gray', zorder=1)
    ax1.legend()
    ax1.set_title('Loss')
    ax2.plot(range(len(accs)), accs, color=colors[0], lw=.5, label='Training set accuracy by batch', zorder=2)
    ax2.scatter(xs, h.data['training_accuracy'], color=colors[1], edgecolors='black',
                label='Training set accuracy by epoch', zorder=2)
    ax2.scatter(xs, h.data['test_accuracy'], color=colors[2], edgecolors='black', label='Test set accuracy by epoch',
                zorder=2)
    for x in xs:
        ax2.axvline(x, lw=1, color='gray', zorder=1)
    ax2.legend()
    ax2.set_title('Accuracy')
    plt.suptitle(title)
    plt.show()

def plot_label_dist(predictions_probs, predicted_class, true_label):
    plt.figure(figsize=(4, 4))
    thisplot = plt.bar(range(10), predictions_probs, color="#77aaaa")
    plt.ylim([0, 1])
    thisplot[predicted_class].set_color(colors[0])
    thisplot[true_label].set_edgecolor(colors[4])
    thisplot[true_label].set_linewidth(1)
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    plt.show()