import numpy as np
import requests
import sys

X = np.load("data/X_test_normal.npy")
i = np.random.randint(0, len(X))
url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/infer"
resp = requests.post(url, json={"window": X[i].tolist()})
print(resp.json())