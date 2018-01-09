import matplotlib.pyplot as plt
from CuteFlower2.data_loading import cd
import os


def save_hist_plot(history, name="test", path=None):

    train_errors = history.history['loss']
    val_errors = history.history['val_loss']

    plt.style.use('bmh')
    plt.plot(range(len(train_errors)), train_errors, 'g-', label="Train")
    plt.plot(range(len(val_errors)), val_errors, 'r-', label="Val")
    plt.legend()

    if path is None:
        path = os.getcwd()+"/Data"

    with cd(path):
        plt.savefig("Train_val_graph_{}".format(name))
        plt.clf()


def intermediate_drawer(name, path=None, draw=False):
    train_loss = []
    val_loss = []
    plt.style.use('bmh')

    def drawer(logs):

        train_loss.append(logs['loss'])
        val_loss.append(logs['val_loss'])

        loss_range = range(len(train_loss))

        plt.ion()  # Ved ikke om man skal gøre det i hvert loop, det er nok fint at have den udenfor men w/e

        train_loss_plot, = plt.plot(
            loss_range, train_loss, label='Training Loss')
        val_loss_plot, = plt.plot(
            loss_range, val_loss, label='Validation loss')

        plt.legend(handles=[train_loss_plot, val_loss_plot])

        if not draw:
            plt.show()
            plt.pause(0.001)

        if path is not None:
            with cd(path):
                plt.savefig("Train_val_graph_{}".format(name))

        plt.clf()

    return drawer
