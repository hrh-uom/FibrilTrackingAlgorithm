import numpy as np
import threading, time, multiprocessing, os
import glob
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib import animation
import customFunctions as md
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
from tqdm import tqdm
import a0_initialise as a0

plt.style.use('./mystyle.mplstyle')

print("b2: Volume rendering")

def load_FTA_results():
    d, MC, props = a0.initialise_dataset()
    md.create_Directory(d.dirOutputs+'/volumerendering')
    try:
        # path=glob.glob( d.dirOutputs + 'fib_rec*'+f'{d.frac*100}*')[0];
        path=glob.glob( d.dirOutputs + 'fib_rec_trim*')[0]
        FR=np.load(path) #original, import fibril record
        nF=FR.shape[0]
        print(f'nF={nF}')
    except:
        FR=0
        print("Error, no fibrec found")
    try:
        path=glob.glob( d.dirOutputs + f'label*')[0]
        volume=np.load(path) #original, import fibril record
        # if np.any(junk <=d.nP): #CORRECTING JUNK PLANES
        #     volume=np.delete(volume,junk[junk<=d.nP], axis=0) #deletes junk planes
    except:
        volume=0
        print("Error, no labelled volume found")
    return d, MC, props, FR, volume, nF
#%%
def volume_render(labels, d, z1, z2, x1, x2,filename,resamplex=1, resamplez=1,el=40,aspect=True, show=True):
    minivol=labels[z1:z2:resamplez, x1:x2:resamplex, x1:x2:resamplex]#resampled volume
    print("Volume rendering image")
    fig = plt.figure() ; ax = fig.add_subplot(projection='3d')
    whichfibs=np.unique(minivol)[np.unique(minivol)>0] ; j=0
    for i in tqdm(whichfibs):
        minivol_coords=np.argwhere(minivol==i)
        ax.scatter(resamplex*d.pxsize*minivol_coords[:,1]/1000, resamplex*d.pxsize*minivol_coords[:,2]/1000, resamplez*d.dz*minivol_coords[:,0]/1000, marker=',')
    proportions=(resamplex*d.pxsize*minivol.shape[1],resamplex*d.pxsize*minivol.shape[2],resamplez*d.dz*minivol.shape[0])
    print(f'ax type {type(ax)}')
    try:
        if aspect:
            #try set_aspect
            # ax.set_aspect(proportions)
            ax.set_box_aspect(proportions)  # aspect ratio is 1:1:1 in data space
    except:
        dummy=0
        print("unable to set box aspect")
    ax.view_init(elev=el, azim=225)
    ax.set_xlabel('x ($\mu$m)', labelpad=10);ax.set_zlabel('z ($\mu$m)', labelpad=10);ax.set_ylabel('y ($\mu$m)', labelpad=10)
    padz=10;padx=0;ax.tick_params(axis='x', pad=padx);ax.tick_params(axis='y', pad=padx);ax.tick_params(axis='z', pad=padz)
    print("Saving VR rendering image"); plt.savefig(d.dirOutputs+filename+f'dummy'+'.png')
    if show:
        plt.show()

#%%-----------MAIN FLOW
if __name__=='__main__':
    # parallell_process()
    d, MC, props, FR, volume, nF=load_FTA_results()
    volume_render(volume, d, 0, d.nP, 0, d.npix, 'volumerendering/volume-render', resamplex=20, resamplez=20, show=False)
    import os, psutil; print('memory ') ; print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
