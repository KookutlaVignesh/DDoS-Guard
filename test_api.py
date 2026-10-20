import requests
import json
import uuid

# First generate a sample csv
import pandas as pd
import numpy as np

edus = ['Bachelors','Masters','HS-grad','Some-college','Doctorate']
cats = ['A','B','C']
data = []
for i in range(300):
    data.append({
        'age': np.nan if i%7==0 else np.random.randint(20, 65),
        'income': np.nan if i%11==0 else np.random.randint(20000, 100000),
        'education': np.random.choice(edus),
        'hours_per_week': np.nan if i%9==0 else np.random.randint(25, 55),
        'score': round(np.random.rand()*100, 1),
        'category': cats[i%3],
        'target': '<=50K' if i%2==0 else '>50K'
    })
df = pd.DataFrame(data)
df.to_csv('test_sample.csv', index=False)

session = requests.Session()
# Upload
with open('test_sample.csv', 'rb') as f:
    res = session.post('http://127.0.0.1:5000/api/upload', files={'file': f})
print("Upload status:", res.status_code, res.text[:100])

# Preprocess (fill missing)
res = session.post('http://127.0.0.1:5000/api/preprocess', json={"step": "fill_median", "col": ""})
res = session.post('http://127.0.0.1:5000/api/preprocess', json={"step": "fill_mode", "col": ""})

# Run CLAPP
res = session.post('http://127.0.0.1:5000/api/clapp', json={
    "target": "target",
    "k": 1,
    "folds": 5,
    "sigma": 0.5,
    "threshold": 0.64
})
print("CLAPP status:", res.status_code, res.text[:200])
