
import matplotlib.pyplot as plt


def plot_history(history:tuple):
    """
    Plot loss and accuracy curves of train and validation process.

    Args:
        history (tuple of lists): ([training loss], [training acc], [validation loss], [validation acc])
    """

    plt.subplot(121)
    plt.plot(history[0], label="training loss")
    plt.plot(history[2], label="validation loss")
    plt.legend(loc='best')
    plt.subplot(122)
    plt.plot(history[1], label="training acc")
    plt.plot(history[3], label="validation acc")
    plt.legend(loc='best')
    plt.show()
