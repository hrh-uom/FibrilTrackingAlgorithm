import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import customFunctions as md
import glob
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#----------------------------------------------------------------------------
#.....................................USER INPUT.............................
#-------------------------------------------------------------------------------

start_plane, end_plane= 0, 101

#----------------------------------------------------------------------------
#....................LOAD DATA FROM FIBRIL MAPPING....................
#------------------------------------------------------------------------------
parent_dir='/Users/user/Dropbox (The University of Manchester)/fibril-tracking/toy-data/'
dir3V=parent_dir+'fin/'# find the relevant data based on the timepoint desired
junk=pd.read_csv( parent_dir+'junkslices.csv', header=None).to_numpy().T[0]-start_plane #which slices are broken
pxsize, dz=np.genfromtxt( parent_dir+'pxsize.csv', delimiter=',')[1] #import voxel size
dirResults= parent_dir+f'results_{start_plane}_{end_plane}/'
MC=np.load(dirResults+'morphComp.npy')
props=np.load(dirResults+'props.npy')
nplanes, npix, _=MC.shape

try:
    path=glob.glob( dirResults + 'fib_rec_trim*')[0];
    FR=np.load(path) #original, import fibril record
except:
    print("Error, no fibrec found")

try:
    path=glob.glob( dirResults + 'label*')[0];
    volume=np.load(path).astype(float) #original, import fibril record
except:
    print("Error, no labelled volume found")


nfibs=FR.shape[0]
volume[tuple(junk[junk<=nplanes])]=np.nan #sets junk planes to nan
volume=np.delete(volume,junk[junk<=nplanes], axis=0) #deletes junk planes

minivol=volume[:, 300:350, 300:350]#subsection
minivol=volume
#%%----------------------------------------------------------------------------
#...................VOLUME RENDERING ...................
#------------------------------------------------------------------------------

#plotting
fig = plt.figure(figsize=(8,20))
ax = Axes3D(fig)

whichfibs=np.unique(minivol)[np.unique(minivol)>0]

for i in whichfibs:
    minivol_coords=np.argwhere(minivol==i)
    # print(minivol_coords)
    ax.scatter(pxsize*minivol_coords[:,1], pxsize*minivol_coords[:,2], dz*minivol_coords[:,0], marker=',')

proportions=(pxsize*minivol.shape[1],pxsize*minivol.shape[2],dz*minivol.shape[0])
ax.set_box_aspect(proportions)  # aspect ratio is 1:1:1 in data space
ax.view_init(elev=30, azim=225)
ax.set_xlabel('x (nm)');ax.set_zlabel('z (nm)');ax.set_ylabel('y (nm)')
dirResults
plt.savefig(dirResults+'3d-render');plt.show()
os.system('say "fuck this, im done."')
