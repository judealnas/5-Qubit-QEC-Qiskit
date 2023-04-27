# %%
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

# %%
datafiles = ['qerr_single_step_sweep.npz',
             'eff_single_step_sweep.npz',
             'dc_single_step_sweep.npz']
             

datas = [np.load(f) for f in datafiles]
figs,axs = zip(*[plt.subplots() for f in datafiles])

hs = [0]*len(datas)

for i in range(len(datas)):
    X = datas[i]['row_data']
    Y = datas[i]['col_data']
    X,Y = np.meshgrid(X,Y)
    Z = datas[i]['fids']

    hs[i] = axs[i].pcolormesh(X,Y,Z,cmap='seismic',antialiased=True)

axs[0].set_xlabel('$p_x$')
axs[0].set_ylabel('$p_z$')

axs[1].set_xlabel('$p$')
axs[1].set_ylabel(r'$\eta$')

axs[2].set_xlabel('$p$')
axs[2].set_ylabel(r'$\delta$')

for (fig,h) in zip(figs,hs):
    fig.colorbar(h) 

plt.show()
# %%
