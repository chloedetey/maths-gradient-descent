import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Pour les graphes 3D

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
    
def plot_surface_3d(f, xlim=(-5, 5), ylim=(-5, 5), title="Surface 3D"):
    """
    Affiche la fonction en 3D, pratique pour voir la forme globale et repérer visuellement où sont les minima.
    """
    # On crée une grille de points x,y
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    
    # On calcule f pour chaque point de la grille
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    
    # Création de la figure 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # On trace la surface avec un dégradé de couleurs
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    ax.set_title(title)
    
    return fig, ax