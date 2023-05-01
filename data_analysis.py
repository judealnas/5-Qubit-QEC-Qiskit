# %%
import os
import sys
sys.path.append(os.path.abspath('./Sim Scripts/'))
import glob
import re
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

states = ['0','r','+']
states_dict = {'0':'0', '+':'+', 'r':'i'}
param_prefixes = ['quant','meas','eff','dc']
label_dict = {'px':'$p_x$', 'pz':'$p_z$', 'p': '$p$',
               'eff': r'$\eta$', 'dc': r'$\delta$' }
# %% Plots of sweeps
%clear
for state in states:
    for prefix in param_prefixes:
        filename = f'{prefix}_err_sweep_{state}.npz'
        npz = np.load(filename)
        
        X = npz['row_data']
        Y = npz['col_data']
        # X,Y = Y,X
        X,Y = np.meshgrid(X,Y)
        
        state_label = state
        x_label = label_dict[npz['row_label'].item()]
        y_label = label_dict[npz['col_label'].item()]

        for fid_type in ['logical','data']:
            fig,ax = plt.subplots(1,1)
            Z = npz[f'fids_{fid_type}']
            h = ax.pcolormesh(Y,X,Z,cmap='seismic')
            fig.colorbar(h) 
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(rf'$\mathcal{{F}}_{state_label}^{{({fid_type[0].upper()})}}$')


plt.show()


# %% Repetition sweep plots
%clear
rep_prefixes = ['eff','dc']

# Regex pattern for extracting quantum error parameter
# from file name
pattern = r'_p([\d.]+)_' 

for prefix in rep_prefixes:
    filenames = glob.glob(f'**/{prefix}_rep_sweep_p*.npz', recursive=True)

    for filename in filenames:
        p = float(re.search(pattern, filename).group(1))
        npz = np.load(filename)
        
        X = npz['row_data']
        Y = npz['col_data']
        # X,Y = Y,X
        X,Y = np.meshgrid(X,Y)
        
        x_label = npz['row_label'].item()
        y_label = npz['col_label'].item()

        for fid_type in ['logical','data']:
            fig,ax = plt.subplots(1,1)
            Z = npz[f'mean_fids_{fid_type}']
            h = ax.pcolormesh(X.T,Y.T,Z,cmap='seismic',antialiased=True,vmin=0,vmax=1)
            fig.colorbar(h) 
            
            ax.set_xticks(npz['row_data'])

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f'$\mathcal{{F}}^{{({fid_type[0].upper()})}}_0\quad; p_{{xz}}={p}$')

# %%
