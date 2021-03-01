import pandas as pd
import pyvista as pv
import numpy as np
from matplotlib.colors import ListedColormap
import customFunctions as md
import glob
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------
#.....................................USER INPUT.............................
#-------------------------------------------------------------------------------

whichdata, start_plane, end_plane=0, 10, 25

#----------------------------------------------------------------------------
#....................LOAD DATA FROM FIBRIL MAPPING....................
#------------------------------------------------------------------------------

dir3V=md.find_3V_data(whichdata); # find the relevant data based on the timepoint desired
junk=pd.read_csv( dir3V+'junkslices.csv', header=None).to_numpy().T[0]-start_plane #which slices are broken
pxsize, dz=np.genfromtxt( dir3V+'pxsize.csv', delimiter=',')[1] #import voxel size
dirResults= dir3V+f'results_{start_plane}_{end_plane}\\'
MC=np.load(dirResults+"morphComp.npy")
props=np.load(dirResults+"props.npy")
nplanes, npix, _=MC.shape


try:
    path=glob.glob( dirResults + 'fib_rec_trim_*')[0];
    FR=np.load(path) #original, import fibril record
except:
    print("Error, no fibrec found")

try:
    path=glob.glob( dirResults + 'label*')[0];
    volume=np.load(path).astype(float) #original, import fibril record
except:
    print("Error, no labelled volume found")

FR.shape
np.unique(volume)

nfibs=FR.shape[0]
volume[tuple(junk[junk<=nplanes])]=np.nan #sets junk planes to nan
volume=np.delete(volume,junk[junk<=nplanes], axis=0) #deletes junk planes

#%%

trackedvol=volume.copy()
trackedvol[trackedvol==0]=np.nan #deletes untracked fibrils


tracked_vol = pv.wrap(trackedvol.T)
tracked_vol.spacing = ( pxsize, pxsize, 10*dz)  # These are the cell sizes along each axis


untrackedvol=volume.copy()+1
untrackedvol[untrackedvol>1]=np.nan
untracked_vol = pv.wrap(untrackedvol.T)


untrackedvol.shape
untracked_vol.spacing = ( pxsize, pxsize, 10*dz)  # These are the cell sizes along each axis

p = pv.Plotter(border=True)

my_cmap=np.full((nfibs,4),1)
my_cmap[:,:-1] = np.random.rand(nfibs,3)
#my_cmap[:,:-1] = [1,0,0]
grey = np.array([189/256, 189/256, 189/256, 1])
blue = np.array([12/256, 238/256, 246/256, 1])


p.add_volume(untracked_vol,cmap=ListedColormap(my_cmap), show_scalar_bar =False)
#p.add_volume(untracked_vol,show_scalar_bar =False, cmap=ListedColormap(blue),shade =True)
p.camera_position = [1,1,1]
p.show()
