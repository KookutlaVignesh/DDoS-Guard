import pandas as pd
import numpy as np
from app import prepare_X, encode_y, CLAPP

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 5)
y = np.random.choice(['A', 'B'], 100)

clf = CLAPP(threshold=0.64, sigma=0.5, k=3)
clf.fit(X, y)
preds = clf.predict(X)
print("CLAPP Success! First 5 preds:", preds[:5])
