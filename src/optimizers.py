import numpy as np

def gradient_descent(f, grad_f, x0, learning_rate=0.1, max_iter=1000, tol=1e-6):
    """
    Descente de gradient simple

    f : fonction à minimiser
    grad_f : fonction qui calcule le gradient
    x0 : point initial (np.array)
    learning_rate : pas de descente
    max_iter : nombre max d'itérations
    tol : critère d'arrêt (norme du gradient)
    """
    x = x0.copy() # Ne modifie pas le point initial 
    trajectory = [x.copy()]

    for i in range(max_iter):
        grad = grad_f(x)

        # Critère d'arrêt
        if np.linalg.norm(grad) < tol:
            break

        # Mise à jour
        x = x - learning_rate * grad
        trajectory.append(x.copy())

    return x, np.array(trajectory) # x : solution finale , trajectory : le chemin suivi 

def gradient_descent_momentum(
    f, grad_f, x0,
    learning_rate=0.1,
    momentum=0.9,
    max_iter=1000,
    tol=1e-6
):
    """
    Descente de gradient avec Momentum
    """
    x = x0.copy()
    v = np.zeros_like(x)
    trajectory = [x.copy()]

    for i in range(max_iter):
        grad = grad_f(x)

        if np.linalg.norm(grad) < tol:
            break

        # Mise à jour de la vitesse (momentum)
        v = momentum * v + learning_rate * grad

        # Mise à jour de la position
        x = x - v
        trajectory.append(x.copy())

    return x, np.array(trajectory)

def gradient_descent_nesterov(
    f, grad_f, x0,
    learning_rate=0.1,
    momentum=0.9,
    max_iter=1000,
    tol=1e-6
):
    """
    Descente de gradient avec Nesterov (NAG)
    """
    x = x0.copy()
    v = np.zeros_like(x)
    trajectory = [x.copy()]

    for i in range(max_iter):
        # Point anticipé
        x_lookahead = x - momentum * v

        grad = grad_f(x_lookahead)

        if np.linalg.norm(grad) < tol:
            break

        v = momentum * v + learning_rate * grad
        x = x - v
        trajectory.append(x.copy())

    return x, np.array(trajectory)

def gradient_descent_adam(
    f, grad_f, x0,
    learning_rate=0.1,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    max_iter=1000,
    tol=1e-6
):
    """
    Descente de gradient avec Adam
    """
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    trajectory = [x.copy()]

    for t in range(1, max_iter + 1):
        grad = grad_f(x)

        if np.linalg.norm(grad) < tol:
            break

        # Moments
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        # Correction du biais
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Mise à jour
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        trajectory.append(x.copy())

    return x, np.array(trajectory)

