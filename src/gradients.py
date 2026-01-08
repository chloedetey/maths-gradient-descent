import numpy as np

""" Méthode 1 : Gradient numérique """

def gradient_numerique(f, x, h=1e-5):
    """
    Calcule le gradient numérique de f en x par différences finies
    Approximation de la dérivée avec f'(x) = (f(x+h) - f(x)) / h
    C'est la pente entre deux points très proches

    Paramètres:
    f : fonction à dériver, f(x)
    x : point où calculer le gradient, np.array([x, y])
    h : petit pas

    Retourne:
    gradient : np.array([df/dx, df/dy])
    """
    grad = np.zeros_like(x)

    # dérivée par rapport à x
    x_plus_h = x.copy()
    x_plus_h[0] += h
    grad[0] = (f(x_plus_h) - f(x)) / h

    # dérivée par rapport à y
    y_plus_h = x.copy()
    y_plus_h[1] += h
    grad[1] = (f(y_plus_h) - f(x)) / h

    return grad

""" Méthode 2 : nombres duaux, dérivation automatique """

class DualNumber:
    """
    Nombre dual : a + b*ε  où ε² = 0
    Quand on calcule f(x + ε), on obtient f(x) + f'(x)*ε
    La partie duale contient automatiquement la dérivée
    """

    def __init__(self, real, dual=0.0):
        self.real = real
        self.dual = dual
    
    def __repr__(self):
        """ Afficher joliment le nombre dual"""
        return f"({self.real} + {self.dual}ε)"

    def __add__(self, other):
        """ Addition : (a + bε) + (c + dε) = (a+c) + (b+d)ε """
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real,
                              self.dual + other.dual)
        else:
            return DualNumber(self.real + other, self.dual)

    __radd__ = __add__

    def __sub__(self, other):
        """ Soustraction : (a + bε) - (c + dε) = (a-c) + (b-d)ε """
        if isinstance(other, DualNumber):
            return DualNumber(self.real - other.real,
                              self.dual - other.dual)
        else:
            return DualNumber(self.real - other, self.dual)

    def __rsub__(self, other):
        return DualNumber(other - self.real, -self.dual)

    def __mul__(self, other):
        """ Multiplication : (a + bε) * (c + dε) = ac + (ad + bc)ε """
        if isinstance(other, DualNumber):
            return DualNumber(self.real * other.real,
                              self.real * other.dual + self.dual * other.real)
        else:
            return DualNumber(self.real * other, self.dual * other)

    __rmul__ = __mul__

    def __pow__(self, power):
        """ Puissance : (a + bε)^n = a^n + n * a^(n-1) * b * ε """
        return DualNumber(
            self.real ** power,
            power * self.real ** (power - 1) * self.dual
        )

    """ Fonctions mathématiques """

    def exp(self):
        """ Exponentielle : dérivée de exp(x) = exp(x)"""
        e = np.exp(self.real)
        return DualNumber(e, e * self.dual)

    def sin(self):
        """ Sinus : dérivée de sin(x) = cos(x)"""
        return DualNumber(np.sin(self.real),
                          np.cos(self.real) * self.dual)
    
    def cos(self):
        """ Cosinus : dérivée de cos(x) = -sin(x)"""
        return DualNumber(np.cos(self.real),
                          -np.sin(self.real) * self.dual)
    def sqrt(self):
        """ Racine carrée : dérivée de sqrt(x) = 1/(2*sqrt(x))"""
        s = np.sqrt(self.real)
        return DualNumber(s, self.dual / (2 * s))
    
    def log(self):
        """ Logarithme : dérivée de log(x) = 1/x"""
        return DualNumber(np.log(self.real),
                          self.dual / self.real)
    
""" Fonctions utilitaires pour utiliser exp, sin, cos avec DualNumber ou float """

def dual_exp(x):
    if isinstance(x, DualNumber):
        return x.exp()
    return np.exp(x)

def dual_sin(x):
    if isinstance(x, DualNumber):
        return x.sin()
    return np.sin(x)

def dual_cos(x):
    if isinstance(x, DualNumber):
        return x.cos()
    return np.cos(x)

def dual_sqrt(x):
    if isinstance(x, DualNumber):
        return x.sqrt()
    return np.sqrt(x)     

""" Fonction gradient avec dual numbers """

def gradient_dual(f, x):
    """
    Calcule le gradient de f en x avec les nombres duaux

    - Pour trouver df/dx : on met ε sur x (dual=1) et pas sur y (dual=0)
    - Pour trouver df/dy : on met ε sur y et pas sur x
    - La partie "dual" du résultat contient la dérivée

    Note : la fonction f doit utiliser les opérations compatibles avec DualNumber (fonctions utilitaires) pour Ackley
    """
    grad = np.zeros_like(x, dtype=float) # dtype=float pour éviter les erreurs

    # Dérivée par rapport à x
    x_dual = [DualNumber(x[0], 1.0), DualNumber(x[1], 0.0)]
    result = f(x_dual)
    grad[0] = result.dual

    # Dérivée par rapport à y
    y_dual = [DualNumber(x[0], 0.0), DualNumber(x[1], 1.0)]
    result = f(y_dual)
    grad[1] = result.dual

    return grad
