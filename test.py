import numpy as np
from src.functions import booth
from src.gradients import gradient_numerique
from src.optimizers import gradient_descent

x0 = np.array([-4.0, 3.0])
grad_f = lambda x: gradient_numerique(booth, x)
sol, traj = gradient_descent(booth, grad_f, x0, learning_rate=0.1, max_iter=100)

print(f"Solution : {sol}")
print(f"Attendu : [1, 3]")
print(f"Itérations : {len(traj)}")