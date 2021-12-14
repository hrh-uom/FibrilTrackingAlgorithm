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
from IPython.display import HTML
plt.style.use('./mystyle.mplstyle')

#----------------------------------------------------------------------------
#.........................USER INPUT.............................
#-------------------------------------------------------------------------------
start_plane, end_plane=0,695;frac=0.14 #corresponds to 1000nm

dirResults, dir3V=md.getDirectories(start_plane, end_plane)
md.create_Directory(dirResults+'/volumerendering')
#----------------------------------------------------------------------------
#....................LOAD DATA FROM FIBRIL MAPPING....................
#------------------------------------------------------------------------------

try:
    path=glob.glob( dirResults + 'morphComp*')[0];
    MC=np.load(path) #original, import fibril record
except:
    print("Error, no MC found")
try:
    path=glob.glob( dirResults + 'fib_rec*'+f'{frac}*')[0];
    FR=np.load(path) #original, import fibril record
except:
    print("Error, no fibrec found")
try:
    path=glob.glob( dirResults + f'label*{frac}*')[0];
    volume=np.load(path).astype(float) #original, import fibril record
except:
    print("Error, no labelled volume found")

#----------------------------------------------------------------------------
#....................METADATA AND DIMENSIONS....................
#------------------------------------------------------------------------------
#Read Metadata File
meta_frame=pd.read_csv(glob.glob(dir3V+'/*metadata*csv')[0])
pxsize=meta_frame.pixelsize[0];junk=meta_frame.junkslices; dz=meta_frame.dz[0]
nplanes, npix, _=MC.shape
nfibs=FR.shape[0]
volume.shape
if np.any(junk <=nplanes): #CORRECTING JUNK PLANES
    volume=np.delete(volume,junk[junk<=nplanes], axis=0) #deletes junk planes

#%%---------------------------------------------------------------------------
#...................STEP THROUGH ANIMATION ...................
#--------------------------------------------------------------------------
# md.export_animation(dirResults, np.arange(nfibs) ,volume,dt=20,step=2)
#%%
# #TRYING TO MAKE OWN RGB LABEL
# import importlib
# importlib.reload(md);
#
# # labels=md.label_volume(MC, np.arange(nfibs), FR, 110, 100)
# # junk
# # md.animation_inline(np.arange(nfibs) ,labels, step=1)
#
#
# labels=md.label_volume(MC, np.arange(nfibs), FR, 10)
#
# nfibs=np.max(labels)+1
# fib_group=np.unique(labels)
# fib_group
# labels.shape
# color = [tuple(np.random.random(size=3).astype('float16')) for i in range(nfibs)]
# color.insert(0, (1.,1.,1.))
# pID=0
# labels.shape[1]
#
# rgblabel=np.zeros((labels.shape[1],labels.shape[2], 3))
# for i in fib_group:
#     #includes -1
#     rgblabel[np.argwhere(labels[pID]==0)]=color[i]
#
# plt.imshow(rgblabel)
#
# labels[pID]

#%%----------------------------------------------------------------------------
#...................MID -PLANE IMAGE  ...................
#--------------------------------------------------------------------------
def midPlaneImage():
    """
    Creates an image of the mapping halfway through the volume
    """
    fig, ax=plt.subplots(figsize=(12, 10))
    # color = [tuple(np.random.random(size=3)) for i in range(int(np.max(volume)))] #randomcolourlist
    choice=tuple(np.array([255, 40,40])/255)
    color = [choice for i in range(int(np.max(volume)))]
    color.insert(0,(.5,.5, .5)) #makesure other fibrils are white!!
    ax.set_xlabel('x ($\mu$m)');ax.set_ylabel('y ($\mu$m)')
    plt.imshow(label2rgb(volume[int(nplanes/2)], bg_label=-1,colors=color), extent=[0,npix*pxsize/1000,0, npix*pxsize/1000])
    plt.savefig(dirResults+'volumerendering/mid-stack-img');
    plt.show()
midPlaneImage()

#%%----------------------------------------------------------------------------
#...................VOLUME RENDERING ...................
#------------------------------------------------------------------------------
md.volume_render(volume, 0, nplanes,0 ,npix, pxsize, dz, dirResults, '/volumerendering/3d-render', el=50 )

#%%




#%%----------------------------------------------------------------------------
#...................VOLUME RENDERING ANIMATION SANDBOX...................
#------------------------------------------------------------------------------
"""

minivol=volume[:, 300:350, 300:350]#subsection
# minivol=volume

#plotting
fig = plt.figure(figsize=(12, 10))
ax=fig.add_subplot(projection='3d')

whichfibs=np.unique(minivol)[np.unique(minivol)>0]

for i in whichfibs:
    minivol_coords=np.argwhere(minivol==i)
    # print(minivol_coords)
    ax.scatter(pxsize*minivol_coords[:,1], pxsize*minivol_coords[:,2], dz*minivol_coords[:,0], marker=',')
    ax.set_box_aspect(proportions)  # aspect ratio is 1:1:1 in data space
    ax.view_init(elev=30, azim=225)
    ax.set_xlabel('x (nm)');ax.set_zlabel('z ($\mu$m)');ax.set_ylabel('y ($\mu$m)')

ax.set_box_aspect(proportions)  # aspect ratio is 1:1:1 in data space
ax.view_init(elev=30, azim=225)
ax.set_xlabel('x (nm)');ax.set_zlabel('z ($\mu$m)');ax.set_ylabel('y ($\mu$m)')
plt.show()
#%%

fig = plt.figure(figsize=(12, 10))
ax=fig.add_subplot(projection='3d')
container = []
import scipy.ndimage
# labels=label_volume(morphComp,fib_group, fib_rec, endplane,startplane)
# color = [tuple(np.random.random(size=3)) for i in range(len(fib_group))] #randomcolourlist
# color.insert(0,(1.,1.,1.)) #makesure other fibrils are white!!
# rgblabel=label2rgb(labels, bg_label=-1, colors=color);
for pID in range(1, nplanes):

    print(f'animating {pID}')
    minivol=volume[0:pID, 0:300, 0:300]
    for i in whichfibs:
        minivol_coords=np.argwhere(minivol==i)
        im=ax.scatter(pxsize*minivol_coords[:,1], pxsize*minivol_coords[:,2], dz*minivol_coords[:,0], marker=',')
    ax.set_box_aspect(proportions)  # aspect ratio is 1:1:1 in data space
    ax.view_init(elev=30, azim=225)
    ax.set_xlabel('x (nm)');ax.set_zlabel('z ($\mu$m)');ax.set_ylabel('y ($\mu$m)')
    proportions=(pxsize*minivol.shape[1],pxsize*minivol.shape[2],dz*minivol.shape[0])
    ax.set_box_aspect(proportions)  # aspect ratio is 1:1:1 in data space
    ax.view_init(elev=30, azim=225)
    ax.set_xlabel('x (nm)');ax.set_zlabel('z ($\mu$m)');ax.set_ylabel('y ($\mu$m)')
    container.append([im])

ani = animation.ArtistAnimation(fig, container, interval=1000, blit=True)
plt.close();
HTML(ani.to_html5_video())
"""
