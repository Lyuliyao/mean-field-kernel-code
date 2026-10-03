import numpy as np
data = np.load("./data/simulation_1.npy")
for i in range(1,101):
    data = np.concatenate((data, np.load(f"./data/simulation_{i}.npy")), axis=1)
np.save("result.npy",data)