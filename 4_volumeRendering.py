import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib import animation
import customFunctions as md
import glob
import os
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('./mystyle.mplstyle')

#----------------------------------------------------------------------------
#.........................USER INPUT.............................
#-------------------------------------------------------------------------------

start_plane, end_plane= 0, 101
parent_dir='/Users/user/Dropbox (The University of Manchester)/fibril-tracking/toy-data/'
dirResults= parent_dir+f'results_{start_plane}_{end_plane}/'

#----------------------------------------------------------------------------
#....................LOAD DATA FROM FIBRIL MAPPING....................
#------------------------------------------------------------------------------

try:
    path=glob.glob( dirResults + 'morphComp*')[0];
    MC=np.load(path) #original, import fibril record
except:
    print("Error, no fibrec found")

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



#----------------------------------------------------------------------------
#....................METADATA AND DIMENSIONS....................
#------------------------------------------------------------------------------
pxsize, dz=np.genfromtxt( parent_dir+'pxsize.csv', delimiter=',')[1]/1000 #import voxel size
nplanes, npix, _=MC.shape
nfibs=FR.shape[0]
junk=pd.read_csv( parent_dir+'junkslices.csv', header=None).to_numpy().T[0]-start_plane #which slices are broken

volume[tuple(junk[junk<=nplanes])]=np.nan #sets junk planes to nan
volume=np.delete(volume,junk[junk<=nplanes], axis=0) #deletes junk planes

#----------------------------------------------------------------------------
#...................MID -PLANE IMAGE  ...................
#--------------------------------------------------------------------------
fig, ax=plt.subplots(figsize=(12, 10))
color = [tuple(np.random.random(size=3)) for i in range(int(np.max(volume)))] #randomcolourlist
color.insert(0,(1.,1.,1.)) #makesure other fibrils are white!!
ax.set_xlabel('x ($\mu$m)');ax.set_ylabel('y ($\mu$m)')
plt.imshow(label2rgb(volume[int(nplanes/2)], bg_label=-1,colors=color), extent=[0,npix*pxsize,0, npix*pxsize])
plt.savefig(dirResults+'mid-stack-img');plt.show()

#%%----------------------------------------------------------------------------
#...................VOLUME RENDERING ...................
#------------------------------------------------------------------------------

minivol=volume[:, 300:350, 300:350]#subsection
# minivol=volume

#plotting
fig = plt.figure(figsize=(12, 10))
ax = Axes3D(fig)

whichfibs=np.unique(minivol)[np.unique(minivol)>0]

for i in whichfibs:
    minivol_coords=np.argwhere(minivol==i)
    # print(minivol_coords)
    ax.scatter(pxsize*minivol_coords[:,1], pxsize*minivol_coords[:,2], dz*minivol_coords[:,0], marker=',')

proportions=(pxsize*minivol.shape[1],pxsize*minivol.shape[2],dz*minivol.shape[0])
ax.set_box_aspect(proportions)  # aspect ratio is 1:1:1 in data space
ax.view_init(elev=30, azim=225)
ax.set_xlabel('x (nm)');ax.set_zlabel('z ($\mu$m)');ax.set_ylabel('y ($\mu$m)')
dirResults
plt.savefig(dirResults+'3d-render');plt.show()

#%%----------------------------------------------------------------------------
#...................VOLUME RENDERING ANIMATION...................
#------------------------------------------------------------------------------


minivol=volume[:, 300:350, 300:350]#subsection
# minivol=volume

#plotting
fig = plt.figure(figsize=(12, 10))
ax = Axes3D(fig)

whichfibs=np.unique(minivol)[np.unique(minivol)>0]

for i in whichfibs:
    minivol_coords=np.argwhere(minivol==i)
    # print(minivol_coords)
    ax.scatter(pxsize*minivol_coords[:,1], pxsize*minivol_coords[:,2], dz*minivol_coords[:,0], marker=',')

ax.set_box_aspect(proportions)  # aspect ratio is 1:1:1 in data space
ax.view_init(elev=30, azim=225)
ax.set_xlabel('x (nm)');ax.set_zlabel('z ($\mu$m)');ax.set_ylabel('y ($\mu$m)')
dirResults
plt.savefig(dirResults+'3d-render');plt.show()

#%%
fig = plt.figure(figsize=(12, 10))
ax = Axes3D(fig)
proportions=(pxsize*minivol.shape[1],pxsize*minivol.shape[2],dz*minivol.shape[0])
ax.set_box_aspect(proportions)  # aspect ratio is 1:1:1 in data space
ax.view_init(elev=30, azim=225)
ax.set_xlabel('x (nm)');ax.set_zlabel('z ($\mu$m)');ax.set_ylabel('y ($\mu$m)')
minivol=volume[:, 300:350, 300:350]#subsection
ax.scatter(pxsize*minivol_coords[:,1], pxsize*minivol_coords[:,2], dz*minivol_coords[:,0], marker=',')
plt.show()

def update(frame_number)

for pID in range(0, nplanes, nplanes//3):

    minivol=volume[0:pID, 300:350, 300:350]#subsection
    whichfibs=np.unique(minivol)[np.unique(minivol)>0]
    for i in whichfibs:
        minivol_coords=np.argwhere(minivol==i)
        # print(minivol_coords)
        im=ax.scatter(pxsize*minivol_coords[:,1], pxsize*minivol_coords[:,2], dz*minivol_coords[:,0], marker=',', animated=True)

    container.append([im])
ani = animation.ArtistAnimation(fig, container, interval=dt, blit=True)
HTML(ani.to_html5_video())

# os.system('say "fuck this, im done."')
