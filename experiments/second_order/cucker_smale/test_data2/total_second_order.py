import numpy as np
data = np.load("./data/simulation_1.npz")
x_data = data["x"]
v_data = data["v"]
for i in range(1,3): # originally range(1,101)
    data = np.load(f"./data/simulation_{i}.npz")
    x_data = np.concatenate((x_data, data["x"]), axis=1)
    v_data = np.concatenate((v_data, data["v"]), axis=1)
np.save("result_second_order.npy",x_data)