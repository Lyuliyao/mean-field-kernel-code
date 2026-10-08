import numpy as np
data = np.load("./data/x_save_second_order_1.npy")
for i in range(1,101):
    data = np.concatenate((data, np.load(f"./data/x_save_second_order_{i}.npy")), axis=1)
np.save("result_second_order.npy",data)