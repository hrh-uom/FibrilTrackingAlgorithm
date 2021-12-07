import numpy as np
import pandas as pd
from math import sqrt
from PIL import Image
import glob
from matplotlib import animation, rc, colors
import matplotlib.image as mpimg
import matplotlib.cm as cm
from skimage.measure import label, regionprops,regionprops_table
from skimage.color import label2rgb
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from  datetime import datetime as dt
import os
plt.style.use('./mystyle.mplstyle')

#----------------DIRECTORIES AND PATHS-------------------
def create_Directory(directory):
    """
    Tests for existence of directory then creates one if not
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def getDirectories(start_plane, end_plane):
    if ('Dropbox' in os.getcwd()):#MY PC
        dirResults=f'/Users/user/Dropbox (The University of Manchester)/fibril-tracking/nuts-and-bolts/csf-output/results_0_695/'
        dir3V='/Users/user/Dropbox (The University of Manchester)/em-images/nuts-and-bolts-3v-data/9am-achilles-fshx-processed/'
    else:#ON CSF
        dirResults=f'../nuts-and-bolts/results_{start_plane}_{end_plane}/'
        dir3V='/mnt/fls01-home01/t97721hr/nuts-and-bolts/three-view/'
    create_Directory(dirResults)
    return dirResults, dir3V


#----------------TIMING--------------------
def t_d_stamp():
    return "{:%y-%m-%d_%H-%M}".format(dt.now())
def t_stamp():
    return "{:%H-%M-%S}".format(dt.now())
def beep():
    winsound.Beep(500, 800);

#-------------SEARCH WINDOWS FOR FIBRIL TRACKING --------------
def search_window(cofI, size, npix):
    """
    finds a 'window' to look in for a size of 50 will yield a 100x100px box
    """
    a=np.floor(np.stack( (cofI-[size,size],cofI+[size,size]), axis=0))
    a[a<0]=0;
    a[a>npix]=npix;
    return a.T
def viewwindow(fID,pID, cofI,size, npix, nplanes, morphComp): #View window of searching for error calcs
    xy=search_window(cofI, size, npix)[:,0].tolist();
    recsize=np.ndarray.flatten(np.diff(search_window(cofI, size, npix))).tolist();
    image_label_overlay = label2rgb(morphComp[pID+1], bg_label=0)
    fig, ax = plt.subplots()
    ax.imshow(image_label_overlay, interpolation='nearest')
    plt.title('fID %i. Plane %i of %i. Size %i' % (fID,pID+1, nplanes, size))
    plt.ylabel('y pix')
    plt.xlabel('x pix')
    # Create a Rectangle patch
    rect=patches.Rectangle((xy[0],xy[1]),recsize[0],recsize[1],linewidth=1,edgecolor='w',facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.show()

#-------------TRIM FIBRIL RECORD-------------

def trim_fib_rec(FR_local, MC, dirResults,frac=0.9):
    """
    Trims fibril record to fibrils which are less than some fraction of the total number of planes
    """
    print(f'trimming fibril record to {frac}')
    #Q: How long are all the fibrils in the original fibril rec?
    nfibs, nplanes=FR_local.shape
    nexist=np.zeros(nfibs, dtype='int')
    for i in range(nfibs):
        if i in np.arange(0, nfibs)[0::1000]:
            print (f"Fibril {i} of {nfibs}")
        nexist[i]=np.max(np.nonzero(FR_local[i]>-1))-np.min(np.nonzero(FR_local[i]>-1))+1
    longfibs=np.where(nexist>nplanes*frac)[0]  #the indices of the long fibirls
    #Erasing fibril record for short fibrils. The only way to map between the two is using longfibs. Reindexing also!

    np.save(dirResults+f'fib_rec_trim_{frac:.2f}', FR_local[longfibs])
    labels=label_volume(MC, np.arange(longfibs.size),FR_local[longfibs], nplanes )
    np.save(dirResults+f'labelled_vol_{frac:.2f}', labels)
    return FR_local[longfibs]

#------------------ANIMATION AND VISUALISATION-----------------------------#
def label_volume(morphComp,fib_group,fib_rec,endplane, startplane=0):
    """
    Returns array where each seperate fibril is labelled with a number.
    Same dims as MC input ie nx ny nplanes
    """
    print("Labelling volume")
    labels=np.where(morphComp[startplane:endplane]==0, -1, 0).astype('int16') #turn all fibs to 0, keeping background=-1
    j=0
    for pID in range(startplane, endplane):
        print(f"Labelling volume {pID}") if pID in range(startplane, endplane, (endplane-startplane)//10) else 0
        for i in range (len(fib_group)):
         if fib_rec[fib_group[i], pID]!=-1:
             value=fib_group[i]+1;
             labels[j]=np.where(morphComp[pID]==fib_rec[fib_group[i], pID]+1, value, labels[j])
        j+=1
    return labels
565/5//10/10
import numpy as np
x=1
print('hi') if x in np.arange(0, 565, 565//10) else 0
def volume_render(labels, z1, z2, x1, x2,  pxsize,dz,dirResults,filename, el=40,aspect=True):
    minivol=labels[z1:z2, x1:x2, x1:x2]#subsection
    # minivol=volume
    #plotting
    fig = plt.figure(figsize=(20, 15))
    ax=plt.axes(projection='3d')
    whichfibs=np.unique(minivol)[np.unique(minivol)>0]
    # print(whichfibs)
    j=0
    for i in whichfibs:
        j+=1
        print(f"fibril {j} of {len(whichfibs)}") if j in np.arange(0, len(whichfibs), len(whichfibs//10)) else 0
        minivol_coords=np.argwhere(minivol==i)
        # print(minivol_coords)
        ax.scatter(pxsize*minivol_coords[:,1]/1000, pxsize*minivol_coords[:,2]/1000, dz*minivol_coords[:,0]/1000, marker=',')

    proportions=(pxsize*minivol.shape[1],pxsize*minivol.shape[2],dz*minivol.shape[0])
    str='aspect1'
    if aspect:
        ax.set_box_aspect(proportions)  # aspect ratio is 1:1:1 in data space
        str='aspectSet'

    ax.view_init(elev=el, azim=225)

    ax.set_xlabel('x ($\mu$m)');ax.set_zlabel('z ($\mu$m)');ax.set_ylabel('y ($\mu$m)')
    print("Volume rendering image")
    plt.savefig(dirResults+filename+str+'.png');plt.show()

def create_animation(fib_group, labels, startplane, endplane, dt, fig_size=10, step=1,colourful=True):
    """
    Create mapping animation object
    ORIGINAL THIS ONE WORKS
    """
    nplanes=labels.shape[0]
    if endplane==0:
        endplane=nplanes
    fig, ax=plt.subplots(1, 1, figsize=(fig_size,fig_size))
    container = []
    color = [tuple(np.random.random(size=3)) for i in range(len(fib_group))] #randomcolourlist
    color.insert(0,(1.,1.,1.)) #makesure other fibrils are white!!
    print("Label2RGB")
    rgblabel=label2rgb(labels[startplane:endplane:step, :, :], bg_label=-1, colors=color);
    frameID=0
    for pID in range(startplane, endplane, step):
        print(f"animating plane {pID}")

        im=ax.imshow(rgblabel[frameID], animated=True)
        plot_title = ax.text(0.5,1.05,'Plane %d of %d' % (pID, nplanes),
                 size=plt.rcParams["axes.titlesize"],
                 ha="center", transform=ax.transAxes, )
        container.append([im, plot_title])
        frameID+=1;
    ani = animation.ArtistAnimation(fig, container, interval=dt, blit=True)
    plt.close();
    return ani
    #ani.save(resultsDir+title+'.mp4')

def create_animation_current(fib_group, labels, startplane, endplane, dt, fig_size=10, step=1,colourful=True):
    """
    Create mapping animation object
    this one doesnt work. Something to do with moving the RGB label line
    """
    nplanes=labels.shape[0]
    if endplane==0:
        endplane=nplanes
    fig, ax=plt.subplots(1, 1, figsize=(fig_size,fig_size))
    container = []
    color = [tuple(np.random.random(size=3)) for i in range(len(fib_group))] #randomcolourlist
    color.insert(0,(1.,1.,1.)) #makesure other fibrils are white!!
    frameID=0
    for pID in range(startplane, endplane, step):
        print(f"animating plane {pID}")
        rgblabel=label2rgb(labels[pID, :, :], bg_label=-1, colors=color);
        im=ax.imshow(rgblabel, animated=True)
        plot_title = ax.text(0.5,1.05,'Plane %d of %d' % (pID, nplanes),
                 size=plt.rcParams["axes.titlesize"],
                 ha="center", transform=ax.transAxes, )
        container.append([im, plot_title])
        frameID+=1;
    ani = animation.ArtistAnimation(fig, container, interval=dt, blit=True)
    plt.close();
    return ani
    #ani.save(resultsDir+title+'.mp4')

def export_animation(resultsDir,fib_group, labels, title='volumerendering/stepthrough-animation', startplane=0, endplane=0, dt=500, figsize=20,step=1):
    ani=create_animation(fib_group, labels, startplane, endplane, dt, figsize , step)
    ani.save(resultsDir+title+'.mp4')
def animation_inline(fib_group, labels, startplane=0, endplane=0, dt=500, figsize=10, step=1):
    """
    Animate a certain group of fibrils, given a specific fibril record
    """
    ani=create_animation(fib_group, labels, startplane, endplane, dt, figsize, step)
    return HTML(ani.to_html5_video())

def red_objects_1_plane(obj_group, pID):
 """
 Please feed object numbers (inplane) not fibril numbers
 """
 labels=morphComp[pID].copy()
 for i in range(obj_group.size):
     value=obj_group[i]
     labels=np.where(morphComp[pID]==value+1, -1, labels)
 labels=np.where(labels>0, 1, labels)
 rgblabel=label2rgb(labels, bg_label=0, colors=[(1, 0, 0), (1, 1, 1)])
 plt.imshow(rgblabel)

#---------------PLOTS-------------------
def my_histogram(arr,xlabel, dens=False, title=0, labels=[], cols='g', binwidth=10, xlims=0, pi=False,filename=0, leg=False):
    """
    A histogram, with number on the y axis
    """
    #cols=['red', 'lime', 'blue', 'pink']
    minx=np.min(np.concatenate(arr)) if isinstance(arr,list) else np.min(arr)
    maxx=np.max(np.concatenate(arr)) if isinstance(arr,list) else np.max(arr)
    bins=np.arange(binwidth * np.floor(minx/binwidth),binwidth * (1+np.ceil(maxx/binwidth)),binwidth)
    plt.hist(arr, bins=bins, density=dens, histtype='bar', edgecolor='black', color=cols,label=labels)
    plt.xlabel(xlabel)
    plt.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    ylabel='Density' if dens else 'Number'
    plt.ylabel(ylabel)
    if xlims!=0:
        plt.xlim(xlims)
    if title !=0:
        plt.title(title)
    if pi:
        plt.xticks(np.arange(-np.pi/4, np.pi/2, step=(np.pi/4)), ['-π/4','0','π/4'])
    plt.grid(True)
    if leg:
        plt.legend()
    if filename!=0:
        plt.savefig(filename)
    plt.show()

#------------------------------STATS---------------------

def moving_avg(data, window_width):
    """
    A rolling average over a 1d array
    """
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec
