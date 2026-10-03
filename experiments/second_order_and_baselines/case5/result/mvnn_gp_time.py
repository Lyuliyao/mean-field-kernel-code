import matplotlib.pyplot as plt

f_size = 18

plt.figure()
plt.plot([1000,2000,4000,8000,16000],[1.435000,1.441200,1.447400,1.434500,1.429500], label='MVNN')
plt.plot([1000,2000,4000,8000,16000],[0.060400,0.220800,1.474400,5.978900,23.446800], label='Gausian process model')
plt.legend(fontsize=15)
plt.ylabel('Simulation time (seconds)', fontsize=f_size)
plt.xlabel('N', fontsize=f_size)
plt.tick_params(labelsize=f_size)
plt.xscale('log', base=2)
plt.savefig('mvnn_gp_time.pdf', bbox_inches='tight')
