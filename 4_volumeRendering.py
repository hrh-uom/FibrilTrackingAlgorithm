import pandas as pd
import pyvista as pv
import numpy as np
from matplotlib.colors import ListedColormap
import customFunctions as md
import glob

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
    FR_path=glob.glob( dirResults + 'fib_rec_trim_*')[0];
    FR=np.load(FR_path) #original, import fibril record
except:
    print("Error, no fibrec found")


#%%


FR.shape
MC.shape


volume=md.label_volume(MC,np.arange(FR.shape[0])[::10], FR, nplanes)
data = pv.wrap(volume.T)
data.spacing = ( pxsize, pxsize, 10*dz)  # These are the cell sizes along each axis

# %%Define the colors we want to use
blue = np.array([12/256, 238/256, 246/256, 1])
black = np.array([11/256, 11/256, 11/256, .1])
grey = np.array([189/256, 189/256, 189/256, .01])
yellow = np.array([255/256, 247/256, 0/256, .5])
red = np.array([1, 0, 0, 1])

mapping = np.linspace(0,np.max(FR), 256)
newcolors = np.empty((256, 4))
newcolors[mapping < 10000] = blue
#newcolors[mapping >= 1000] = grey
newcolors[mapping <100]=red

my_colormap = ListedColormap(newcolors)



data.plot(volume=True, cmap=my_colormap, cpos=[100, -100,100]) # Volume render

##%%
p = pv.Plotter(border=True)


#plt.imshow[]


#p.add_mesh(data, color="k")
threshed = data.threshold([1, 10000])
p.add_mesh(threshed,color='red')
p.add_mesh(data.threshold([0, 1]),color='white')
p.camera_position = [-2, 5, 3]
p.show()
