import numpy as np
import pandas as pd
from math import sqrt
from PIL import Image
import glob
from matplotlib import animation, rc, colors
from IPython.display import HTML
import matplotlib.image as mpimg
import matplotlib.cm as cm
from skimage.measure import label, regionprops,regionprops_table
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams['figure.figsize'] = [10, 7.5];#default plot size
plt.rcParams['animation.ffmpeg_path'] = 'C:\\FFmpeg\\bin\\ffmpeg.exe'; # SPECIFIC TO YOUR MACHINE, for inline animations
import winsound #for development
from  datetime import datetime as dt
import os

#----------------DIRECTORIES AND PATHS-------------------
def create_Directory(directory):
    """
    Tests for existence of directory then creates one if not
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
def find_3V_data(whichdata=0):
    """
    finds relevant directory for three view based on choice of timepoint.
    options 9am: 9, 9.15, 9.100
    options 7pm: 19, 19.100
    """
    if whichdata==9:
        dir3View = 'D:\\3View\\9am-achilles-fshx\\7.5um_crop_2\\'
    elif whichdata==19:
        dir3View = 'D:\\3View\\7pmAchx700\\7-5um\\'
    elif whichdata==19.100: #dummyset1!!
        dir3View='D:\\3View\\7pmAchx700\\7-5um\\dummy_0_100\\'
    elif whichdata==9.15: #dummyset1!!
        dir3View='D:\\3View\\9am-achilles-fshx\\7.5um_crop_2\\dummy_110_124\\'
    elif whichdata==9.100:
        dir3View='D:\\3View\\9am-achilles-fshx\\7.5um_crop_2\\dummy_100_200\\'
    else:
        dir3View='C:\\Users\\t97721hr\\Dropbox (The University of Manchester)\\Fibril Tracking Algorithm\\toy_data\\'
    return dir3View

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
    plt.title('fID %i. Plane %i of %i. Size %i' % (fID,pID+1, nplanes))
    plt.ylabel('y pix')
    plt.xlabel('x pix')
    # Create a Rectangle patch
    rect=patches.Rectangle((xy[0],xy[1]),recsize[0],recsize[1],linewidth=1,edgecolor='w',facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.show()

#------------------ANIMATION AND VISUALISATION-----------------------------#
def label_volume(morphComp,fib_group,fib_rec,endplane, startplane=0):
 labels=np.where(morphComp==0, -1, 0) #turn all fibs to 0, keeping background=-1
 for pID in range(startplane, endplane):
     for i in range (len(fib_group)):
         if fib_rec[fib_group[i], pID]!=-1:
             value=fib_group[i]+1;
             labels[pID]=np.where(morphComp[pID]==fib_rec[fib_group[i], pID]+1, value, labels[pID])
 return labels;
def create_animation(morphComp,fib_group, fib_rec, startplane, endplane, dt, fig_size):
    """
    Export mapping animation
    """
    nplanes=morphComp.shape[0]
    if endplane==0:
        endplane=nplanes
    fig, ax=plt.subplots(1, 1, figsize=(fig_size,fig_size))
    container = []
    import scipy.ndimage
    labels=label_volume(morphComp,fib_group, fib_rec, endplane,startplane)
    color = [tuple(np.random.random(size=3)) for i in range(5000)] #randomcolourlist
    color.insert(0,(1.,1.,1.)) #makesure other fibrils are white!!
    rgblabel=label2rgb(labels, bg_label=-1, colors=color);
    for pID in range(startplane, endplane):
        im=ax.imshow(rgblabel[pID], animated=True)
        plot_title = ax.text(0.5,1.05,'Plane %d of %d' % (pID, nplanes),
                 size=plt.rcParams["axes.titlesize"],
                 ha="center", transform=ax.transAxes, )
        container.append([im, plot_title])
    ani = animation.ArtistAnimation(fig, container, interval=dt, blit=True)
    plt.close();
    return ani
    #ani.save(resultsDir+title+'.mp4')
def export_animation(resultsDir, title, morphComp,fib_group, fib_rec, startplane=0, endplane=0, dt=500, figsize=20):
    ani=create_animation(morphComp,fib_group, fib_rec, startplane, endplane, dt, figsize )
    ani.save(resultsDir+title+'.mp4')
def animation_inline(morphComp,fib_group, fib_rec, startplane=0, endplane=0, dt=500, figsize=10):
    """
    Animate a certain group of fibrils, given a specific fibril record
    """
    ani=create_animation(morphComp,fib_group, fib_rec, startplane, endplane, dt, figsize)
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

#---------------STATISTICAL PLOTS-------------------
def my_histogram(arr,xlabel, title='Histogram',  nbins=10):
    """
    A histogram, with number on the y axis
    """
    n, bins, patches = plt.hist(arr, nbins, density=False, facecolor='g', edgecolor='black', alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel('Number')
    plt.title(title)
    plt.grid(True)
    plt.show()
