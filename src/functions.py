#pour les fonctions test 
import numpy as np

##pour des fonctions simples

def quadratique(x, gamma=1.0):
    """
    f(x, y) = x^2 + gamma * y^2
    """
    return x[0]**2 + gamma * x[1]**2
##ici x[0]= X et x[1]= y

def exponentielle(x):
    """
    g(x, y) = 1 - exp(-10 x^2 - y^2)
    """
    return 1 - np.exp(-10 * x[0]**2 - x[1]**2)

def polynomiale(x):
    """
    h(x, y) = x^2 y - 2 x y^3 + 3 x y + 4
    """
    return x[0]**2 * x[1] - 2 * x[0] * x[1]**3 + 3 * x[0] * x[1] + 4

###test avec des fonctions classiques test demandé dans l'énnoncé 

def rosenbrock(x):
    """
    Rosenbrock function
    Minimum global en (1, 1)
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def beale(x):
    """
    Beale function
    Minimum global en (3, 0.5)
    """
    x1, x2 = x[0], x[1]
    return (1.5 - x1 + x1*x2)**2 \
         + (2.25 - x1 + x1*x2**2)**2 \
         + (2.625 - x1 + x1*x2**3)**2


def booth(x):
    """
    Booth function
    Minimum global en (1, 3)
    """
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


def himmelblau(x):
    """
    Himmelblau function
    4 minima globaux
    """
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


def ackley(x):
    """
    Ackley function
    Minimum global en (0, 0)
    """
    x1, x2 = x[0], x[1]
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) \
           - np.exp(0.5 * (np.cos(2*np.pi*x1) + np.cos(2*np.pi*x2))) \
           + np.e + 20

""" Version d'Ackley compatible avec les nombres duaux """
# Utilisée avec gradient_dual

def ackley_dual_compatible(x):
    """
    Ackley pour les nombres duaux.
    Utilise les fonctions de src.gradients pour être compatible.
    """
    from src.gradients import dual_exp, dual_cos, dual_sqrt
    
    x1, x2 = x[0], x[1]
    
    term1 = -20 * dual_exp(-0.2 * dual_sqrt(0.5 * (x1**2 + x2**2)))
    term2 = -dual_exp(0.5 * (dual_cos(2 * 3.14159265359 * x1) + dual_cos(2 * 3.14159265359 * x2)))
    
    return term1 + term2 + 2.71828182846 + 20
           
