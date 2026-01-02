import numpy as np
import matplotlib.pyplot as plt

def plot_contours(f, xlim=(-5, 5), ylim=(-5, 5), levels=50):
    """
    Trace les contours 2D de la fonction f(x, y)
    """
    x = np.linspace(xlim[0], xlim[1], 200)
    y = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    plt.contour(X, Y, Z, levels=levels)
    plt.xlabel("x")
    plt.ylabel("y")

def plot_trajectory(trajectory, color="red", label=None):
    """
    Trace la trajectoire d'un algorithme sur un graphique existant
    """
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], "-o",
             color=color, markersize=3, label=label)