import h5py
import numpy as np
import glob
DT = 1e-2
data_file_list = glob.glob("../DATA_GENERATION4/data/opinion_traj_*")
x_save_list = []
v_save_list = []
for data_file in data_file_list:
    data = np.load(data_file)[...,None]
    print(data_file)
    x = data[:-1]
    v = (data[1:]-data[:-1])/DT
    x_save_list.append(x)
    v_save_list.append(v)
x_save = np.concatenate(x_save_list, axis=0)
v_save = np.concatenate(v_save_list, axis=0)

perms = np.random.permutation(x_save.shape[0])
train_data = {
    "x_save":x_save[perms],
    "v_save":v_save[perms]}
np.savez("./training_data.npz", **train_data)