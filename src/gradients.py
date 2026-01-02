import numpy as np
# Gradient numérique

def gradient_numerique(f, x, h=1e-5):
    """
    Calcule le gradient numérique de f en x par différences finies
    f : fonction f(x)
    x : np.array([x, y])
    h : petit pas
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

# Nombres duaux , dérivée automatique 

class DualNumber:
    def __init__(self, real, dual=0.0):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real,
                              self.dual + other.dual)
        else:
            return DualNumber(self.real + other, self.dual)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real - other.real,
                              self.dual - other.dual)
        else:
            return DualNumber(self.real - other, self.dual)

    def __rsub__(self, other):
        return DualNumber(other - self.real, -self.dual)

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real * other.real,
                              self.real * other.dual + self.dual * other.real)
        else:
            return DualNumber(self.real * other, self.dual * other)

    __rmul__ = __mul__

    def __pow__(self, power):
        return DualNumber(
            self.real ** power,
            power * self.real ** (power - 1) * self.dual
        )


def gradient_dual(f, x):
    """
    Calcule le gradient de f en x avec les nombres duaux
    """
    grad = np.zeros_like(x)

    # dérivée par rapport à x
    x_dual = [DualNumber(x[0], 1.0), DualNumber(x[1], 0.0)]
    grad[0] = f(x_dual).dual

    # dérivée par rapport à y
    y_dual = [DualNumber(x[0], 0.0), DualNumber(x[1], 1.0)]
    grad[1] = f(y_dual).dual

    return grad
