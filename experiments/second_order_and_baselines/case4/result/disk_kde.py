import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time

start = time.time()
result_train = np.load("../sim_second_order_3/result_second_order.npy")
result_test = np.load("../test_data_second_order_3/result_second_order.npy")

plt.figure(figsize=(18,6))
plt.subplots_adjust(wspace=0,hspace=0)
f_size=24
vmax = 0.4
xmax = 3.0
bin_size = 0.1

X, Y = np.mgrid[-xmax:xmax:100j,-xmax:xmax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])

values = np.vstack([result_test[0,:,0],result_test[0,:,1]])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T,X.shape)

ax1 = plt.subplot(2,3,1)
plt.title(r"$t=0$",fontsize=f_size)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().xaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', top=False)
plt.gca().xaxis.set_tick_params(which='minor',size=3,width=2, direction='out', top=False)
plt.gca().yaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', right=False)
plt.gca().yaxis.set_tick_params(which='minor',size=3,width=2, direction='out', right=False)
plt.imshow(np.rot90(Z),cmap="turbo",extent=[-xmax,xmax,-xmax,xmax],vmin=0,vmax=vmax,aspect='auto')
plt.xlim([-xmax,xmax])
plt.ylim([-xmax,xmax])
plt.xticks([-2,0,2],fontsize=24)
plt.yticks([-2,0,2],fontsize=24)
plt.gca().xaxis.set_visible(False) # Hide the y-axis for the third subplot
print('Plot 1 done.')

values = np.vstack([result_test[100,:,0],result_test[100,:,1]])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T,X.shape)

ax2 = plt.subplot(2,3,2)
plt.title(r"$t=1$",fontsize=f_size)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().xaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', top=False)
plt.gca().xaxis.set_tick_params(which='minor',size=3,width=2, direction='out', top=False)
plt.gca().yaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', right=False)
plt.gca().yaxis.set_tick_params(which='minor',size=3,width=2, direction='out', right=False)
plt.imshow(np.rot90(Z),cmap="turbo",extent=[-xmax,xmax,-xmax,xmax],vmin=0,vmax=vmax,aspect='auto')
plt.xlim([-xmax,xmax])
plt.ylim([-xmax,xmax])
plt.xticks([-2,0,2],fontsize=24)
plt.yticks([-2,0,2],fontsize=24)
plt.gca().xaxis.set_visible(False) # Hide the y-axis for the third subplot
plt.gca().yaxis.set_visible(False) # Hide the y-axis for the third subplot
print('Plot 2 done.')

values = np.vstack([result_test[-1,:,0],result_test[-1,:,1]])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T,X.shape)

ax3 = plt.subplot(2,3,3)
plt.title(r"$t=2$",fontsize=f_size)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().xaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', top=False)
plt.gca().xaxis.set_tick_params(which='minor',size=3,width=2, direction='out', top=False)
plt.gca().yaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', right=False)
plt.gca().yaxis.set_tick_params(which='minor',size=3,width=2, direction='out', right=False)
plt.imshow(np.rot90(Z),cmap="turbo",extent=[-xmax,xmax,-xmax,xmax],vmin=0,vmax=vmax,aspect='auto')
plt.xlim([-xmax,xmax])
plt.ylim([-xmax,xmax])
plt.xticks([-2,0,2],fontsize=24)
plt.yticks([-2,0,2],fontsize=24)
plt.gca().xaxis.set_visible(False) # Hide the y-axis for the third subplot
plt.gca().yaxis.set_visible(False) # Hide the y-axis for the third subplot
print('Plot 3 done.')

values = np.vstack([result_train[0,:,0],result_train[0,:,1]])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T,X.shape)

ax4 = plt.subplot(2,3,4,sharex=ax1)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().xaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', top=False)
plt.gca().xaxis.set_tick_params(which='minor',size=3,width=2, direction='out', top=False)
plt.gca().yaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', right=False)
plt.gca().yaxis.set_tick_params(which='minor',size=3,width=2, direction='out', right=False)
plt.imshow(np.rot90(Z),cmap="turbo",extent=[-xmax,xmax,-xmax,xmax],vmin=0,vmax=vmax,aspect='auto')
plt.xlim([-xmax,xmax])
plt.ylim([-xmax,xmax])
plt.xticks([-2,0,2],fontsize=24)
plt.yticks([-2,0,2],fontsize=24)
print('Plot 4 done.')

values = np.vstack([result_train[100,:,0],result_train[100,:,1]])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T,X.shape)

ax5 = plt.subplot(2,3,5,sharex=ax1)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().xaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', top=False)
plt.gca().xaxis.set_tick_params(which='minor',size=3,width=2, direction='out', top=False)
plt.gca().yaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', right=False)
plt.gca().yaxis.set_tick_params(which='minor',size=3,width=2, direction='out', right=False)
plt.imshow(np.rot90(Z),cmap="turbo",extent=[-xmax,xmax,-xmax,xmax],vmin=0,vmax=vmax,aspect='auto')
plt.xlim([-xmax,xmax])
plt.ylim([-xmax,xmax])
plt.xticks([-2,0,2],fontsize=24)
plt.yticks([-2,0,2],fontsize=24)
plt.gca().yaxis.set_visible(False) # Hide the y-axis for the third subplot
print('Plot 5 done.')

values = np.vstack([result_train[-1,:,0],result_train[-1,:,1]])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T,X.shape)

ax6 = plt.subplot(2,3,6,sharex=ax1)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().xaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', top=False)
plt.gca().xaxis.set_tick_params(which='minor',size=3,width=2, direction='out', top=False)
plt.gca().yaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', right=False)
plt.gca().yaxis.set_tick_params(which='minor',size=3,width=2, direction='out', right=False)
h6 = ax6.imshow(np.rot90(Z),cmap="turbo",extent=[-xmax,xmax,-xmax,xmax],vmin=0,vmax=vmax,aspect='auto')
plt.xlim([-xmax,xmax])
plt.ylim([-xmax,xmax])
plt.xticks([-2,0,2],fontsize=24)
plt.yticks([-2,0,2],fontsize=24)
plt.gca().yaxis.set_visible(False) # Hide the y-axis for the third subplot
print('Plot 6 done.')

cbar = plt.colorbar(h6, ax=[ax1, ax2, ax3, ax4, ax5, ax6],
                    ticks=np.linspace(0,vmax,5), pad=0.01)
cbar.ax.tick_params(labelsize=24)
plt.savefig("disk_x.pdf",dpi=600, bbox_inches='tight')
end = time.time()

print(f"Elapsed time: {end - start:.4f} seconds")


#########################################################################


start = time.time()
result_train = np.load("../sim_second_order_3/v_result_second_order.npy")
result_test = np.load("../test_data_second_order_3/v_result_second_order.npy")

plt.figure(figsize=(18,6))
plt.subplots_adjust(wspace=0,hspace=0)
vmax = 0.4
xmax = 2.4
ybin = 2
bin_size = 0.1

X, Y = np.mgrid[-xmax:xmax:100j,-xmax:xmax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])

values = np.vstack([result_test[0,:,0],result_test[0,:,1]])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T,X.shape)

ax1 = plt.subplot(2,3,1)
plt.title(r"$t=0$",fontsize=f_size)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().xaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', top=False)
plt.gca().xaxis.set_tick_params(which='minor',size=3,width=2, direction='out', top=False)
plt.gca().yaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', right=False)
plt.gca().yaxis.set_tick_params(which='minor',size=3,width=2, direction='out', right=False)
plt.imshow(np.rot90(Z),cmap="turbo",extent=[-xmax,xmax,-xmax,xmax],vmin=0,vmax=vmax,aspect='auto')
plt.xlim([-xmax,xmax])
plt.ylim([-xmax,xmax])
plt.xticks([-ybin,0,ybin],fontsize=24)
plt.yticks([-ybin,0,ybin],fontsize=24)
plt.gca().xaxis.set_visible(False) # Hide the y-axis for the third subplot
print('Plot 1 done.')

values = np.vstack([result_test[100,:,0],result_test[100,:,1]])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T,X.shape)

ax2 = plt.subplot(2,3,2)
plt.title(r"$t=1$",fontsize=f_size)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().xaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', top=False)
plt.gca().xaxis.set_tick_params(which='minor',size=3,width=2, direction='out', top=False)
plt.gca().yaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', right=False)
plt.gca().yaxis.set_tick_params(which='minor',size=3,width=2, direction='out', right=False)
plt.imshow(np.rot90(Z),cmap="turbo",extent=[-xmax,xmax,-xmax,xmax],vmin=0,vmax=vmax,aspect='auto')
plt.xlim([-xmax,xmax])
plt.ylim([-xmax,xmax])
plt.xticks([-ybin,0,ybin],fontsize=24)
plt.yticks([-ybin,0,ybin],fontsize=24)
plt.gca().xaxis.set_visible(False) # Hide the y-axis for the third subplot
plt.gca().yaxis.set_visible(False) # Hide the y-axis for the third subplot
print('Plot 2 done.')

values = np.vstack([result_test[-1,:,0],result_test[-1,:,1]])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T,X.shape)

ax3 = plt.subplot(2,3,3)
plt.title(r"$t=2$",fontsize=f_size)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().xaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', top=False)
plt.gca().xaxis.set_tick_params(which='minor',size=3,width=2, direction='out', top=False)
plt.gca().yaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', right=False)
plt.gca().yaxis.set_tick_params(which='minor',size=3,width=2, direction='out', right=False)
plt.imshow(np.rot90(Z),cmap="turbo",extent=[-xmax,xmax,-xmax,xmax],vmin=0,vmax=vmax,aspect='auto')
plt.xlim([-xmax,xmax])
plt.ylim([-xmax,xmax])
plt.xticks([-ybin,0,ybin],fontsize=24)
plt.yticks([-ybin,0,ybin],fontsize=24)
plt.gca().xaxis.set_visible(False) # Hide the y-axis for the third subplot
plt.gca().yaxis.set_visible(False) # Hide the y-axis for the third subplot
print('Plot 3 done.')

values = np.vstack([result_train[0,:,0],result_train[0,:,1]])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T,X.shape)

ax4 = plt.subplot(2,3,4,sharex=ax1)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().xaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', top=False)
plt.gca().xaxis.set_tick_params(which='minor',size=3,width=2, direction='out', top=False)
plt.gca().yaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', right=False)
plt.gca().yaxis.set_tick_params(which='minor',size=3,width=2, direction='out', right=False)
plt.imshow(np.rot90(Z),cmap="turbo",extent=[-xmax,xmax,-xmax,xmax],vmin=0,vmax=vmax,aspect='auto')
plt.xlim([-xmax,xmax])
plt.ylim([-xmax,xmax])
plt.xticks([-ybin,0,ybin],fontsize=24)
plt.yticks([-ybin,0,ybin],fontsize=24)
print('Plot 4 done.')

values = np.vstack([result_train[100,:,0],result_train[100,:,1]])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T,X.shape)

ax5 = plt.subplot(2,3,5,sharex=ax1)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().xaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', top=False)
plt.gca().xaxis.set_tick_params(which='minor',size=3,width=2, direction='out', top=False)
plt.gca().yaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', right=False)
plt.gca().yaxis.set_tick_params(which='minor',size=3,width=2, direction='out', right=False)
plt.imshow(np.rot90(Z),cmap="turbo",extent=[-xmax,xmax,-xmax,xmax],vmin=0,vmax=vmax,aspect='auto')
plt.xlim([-xmax,xmax])
plt.ylim([-xmax,xmax])
plt.xticks([-ybin,0,ybin],fontsize=24)
plt.yticks([-ybin,0,ybin],fontsize=24)
plt.gca().yaxis.set_visible(False) # Hide the y-axis for the third subplot
print('Plot 5 done.')

values = np.vstack([result_train[-1,:,0],result_train[-1,:,1]])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T,X.shape)

ax6 = plt.subplot(2,3,6,sharex=ax1)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().xaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', top=False)
plt.gca().xaxis.set_tick_params(which='minor',size=3,width=2, direction='out', top=False)
plt.gca().yaxis.set_tick_params(which='major',size=6,width=2.5, direction='out', right=False)
plt.gca().yaxis.set_tick_params(which='minor',size=3,width=2, direction='out', right=False)
h6 = ax6.imshow(np.rot90(Z),cmap="turbo",extent=[-xmax,xmax,-xmax,xmax],vmin=0,vmax=vmax,aspect='auto')
plt.xlim([-xmax,xmax])
plt.ylim([-xmax,xmax])
plt.xticks([-ybin,0,ybin],fontsize=24)
plt.yticks([-ybin,0,ybin],fontsize=24)
plt.gca().yaxis.set_visible(False) # Hide the y-axis for the third subplot
print('Plot 6 done.')

cbar = plt.colorbar(h6, ax=[ax1, ax2, ax3, ax4, ax5, ax6],
                    ticks=np.linspace(0,vmax,5), pad=0.01)
cbar.ax.tick_params(labelsize=24)
plt.savefig("disk_v.pdf",dpi=600, bbox_inches='tight')
end = time.time()

print(f"Elapsed time: {end - start:.4f} seconds")
