import numpy as np
x_data = np.load("./data/x_save_second_order_1.npy")
v_data = np.load("./data/v_save_second_order_1.npy")
for i in range(1,101):
    x_data = np.concatenate((x_data, np.load(f"./data/x_save_second_order_{i}.npy")), axis=1)
    v_data = np.concatenate((v_data, np.load(f"./data/v_save_second_order_{i}.npy")), axis=1)
np.save("result_second_order.npy",x_data)
np.save("v_result_second_order.npy",v_data)