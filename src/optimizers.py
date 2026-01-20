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
    x = x0.copy()  # Ne modifie pas le point initial
    trajectory = [x.copy()]

    for i in range(max_iter):
        grad = grad_f(x)

        # Critère d'arrêt
        if np.linalg.norm(grad) < tol:
            break

        # Mise à jour : on descend dans la direction opposée au gradient
        x = x - learning_rate * grad
        trajectory.append(x.copy())

    return x, np.array(trajectory)  # x : solution finale, trajectory : le chemin suivi

def gradient_descent_momentum(
    f, grad_f, x0,
    learning_rate=0.1,
    momentum=0.9,
    max_iter=1000,
    tol=1e-6
):
    """
    Descente de gradient avec Momentum

    On garde l'élan des pas précédents.
    Permet de traverser les plateaux et d'éviter les zigzags dans les ravines.

    v = momentum * v + lr * gradient    on accumule la 'vitesse'
    x = x - v                           on bouge selon cette vitesse
    """
    x = x0.copy()
    v = np.zeros_like(x) # Vitesse initiale = 0
    trajectory = [x.copy()]

    for i in range(max_iter):
        grad = grad_f(x)

        if np.linalg.norm(grad) < tol:
            break

        # Mise à jour de la vitesse (momentum)
        # Le momentum garde une partie de l'ancienne vitesse
        # Le learning rate * grad ajoute la nouvelle direction
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
    Descente de gradient avec Nesterov (NAG : Nesterov Accelerated Gradient)

    Amélioration de Momentum : On calcule le gradient à la position "anticipée".

    x_lookahead = x - momentum * v      où on serait si on continuait
    grad = gradient en x_lookahead      on regarde la pente à cet endroit
    v = momentum * v + lr * grad        on ajuste notre vitesse
    x = x - v                           on bouge
    """
    x = x0.copy()
    v = np.zeros_like(x)
    trajectory = [x.copy()]

    for i in range(max_iter):
        # Point anticipé : où on serait si on continuait avec l'élan actuel
        x_lookahead = x - momentum * v

        # Gradient calculé au point anticipé (pas au point actuel)
        grad = grad_f(x_lookahead)

        if np.linalg.norm(grad) < tol:
            break

        # Mise à jour de la vitesse
        v = momentum * v + learning_rate * grad
        # Mise à jour de la position
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
    Descente de gradient avec Adam (Adaptive Moment Estimation)
    """
    x = x0.copy()
    m = np.zeros_like(x) # Moyenne mobile du gradient (1er moment)
    v = np.zeros_like(x) # Moyenne mobile du gradient^2 (2ème moment)
    trajectory = [x.copy()]

    for t in range(1, max_iter + 1):
        grad = grad_f(x)

        if np.linalg.norm(grad) < tol:
            break

        # Mise à jour des moments
        m = beta1 * m + (1 - beta1) * grad          # Moyenne du gradient
        v = beta2 * v + (1 - beta2) * (grad ** 2)   # Moyenne du gradient^2

        # Correction du biais (important au début quand m et v sont proches de 0)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Mise à jour : le sqrt(v_hat) normalise par dimension
        # Epsilon évite la division par zéro
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        trajectory.append(x.copy())

    return x, np.array(trajectory)

