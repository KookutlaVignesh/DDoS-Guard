import numpy as np
import time

X_train = np.random.rand(40000, 10)
X_test = np.random.rand(10000, 10)

t0 = time.time()
for xr in X_test:
    dists = np.linalg.norm(X_train - xr, axis=1)
    # just do it
t1 = time.time()
print(f"Time: {t1 - t0:.2f}s")
