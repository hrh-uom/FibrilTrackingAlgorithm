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
from tqdm import tqdm
plt.style.use('./mystyle.mplstyle')

#----------------DIRECTORIES AND PATHS-------------------
def create_Directory(directory):
    """
    Tests for existence of directory then creates one if not
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
#----------------TIMING--------------------
def t_d_stamp():
    return "{:%y-%m-%d_%H-%M}".format(dt.now())
def t_stamp():
    return "{:%H-%M-%S}".format(dt.now())
def beep():
    winsound.Beep(500, 800);
def print_status(update, d):
    with open(d.dirOutputs+f'status_update.csv', 'a') as status_update:
        status_update.write(update)
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

def trim_fib_rec(FR_0, MC, dirOutputs,frac, save=True):
    """
    Trims fibril record to fibrils which are less than some fraction of the total number of planes
    """
    print(f'trimming fibril record to {100*frac} percent')
    #Q: How long are all the fibrils in the original fibril rec?
    nfibs, nplanes=FR_0.shape
    nexist=np.zeros(nfibs, dtype='int')
    # print(f'nfibs {nfibs, nplanes, nexist}')
    for i in tqdm(range(nfibs)):
        nexist[i]=np.max(np.nonzero(FR_0[i]>-1))-np.min(np.nonzero(FR_0[i]>-1))+1
    # print(f'nexist {np.unique(nexist)}')
    longfibs=np.where(nexist>nplanes*frac)[0]  #the indices of the long fibirls
    #Erasing fibril record for short fibrils. The only way to map between the two is using longfibs. Reindexing also!
    if save:
        np.save(dirOutputs+f'fib_rec_trim_{100*frac}', FR_0[longfibs])
        labels=label_volume(MC, np.arange(longfibs.size),FR_0[longfibs], nplanes )
        np.save(dirOutputs+f'labelled_vol_{100*frac}', labels)
    return FR_0[longfibs]

#------------------ANIMATION AND VISUALISATION-----------------------------#
def label_volume(morphComp,fib_group,fib_rec,endplane, startplane=0):
    """
    Returns array where each seperate fibril is labelled with a number.
    Same dims as MC input ie nx ny nplanes
    """
    print("Labelling volume")
    labels=np.where(morphComp[startplane:endplane]==0, -1, 0).astype('int16') #turn all fibs to 0, keeping background=-1
    j=0
    for pID in tqdm(range(startplane, endplane)):
        # print(f"Labelling volume {pID}") if pID in range(startplane, endplane, (endplane-startplane)//10) else 0
        for i in range (len(fib_group)):
         if fib_rec[fib_group[i], pID]!=-1:
             value=fib_group[i]+1;
             labels[j]=np.where(morphComp[pID]==fib_rec[fib_group[i], pID]+1, value, labels[j])
        j+=1
    return labels
#
# def custom_RGB_maker(fib_group, labels):
#     """
#     Takes a 3D array of labels and makes each label a different colour in terms of RGB [0, 0, 0] to [255, 255, 255]
#     Custom version of skimage.color.label2rgb which uses int data type to save memory
#     """
#     cols=np.random.randint(0, 255, (np.max(labels)+10, 3), dtype='uint8'); cols[0]=[0,0,0]; cols[1]=[255,255,255]
#     # cols=np.full((np.max(labels)+10, 3), 255, dtype='uint8'); cols[0]=[0,0,0];
#
#     RGB_vol=np.zeros((labels.shape[0], labels.shape[1], labels.shape[2], 3), dtype='uint8')
#     for i in np.arange(labels.shape[0]):
#         RGB_plane=cols[(labels[i]+1)] #https://forum.image.sc/t/skimage-color-label2rgb-but-choose-specific-colors-for-specific-labels/62500
#         RGB_vol[i]=RGB_plane
#     return RGB_vol

def custom_RGB_maker(fib_group, labels):
    """
    Takes a 3D array of labels and makes each label a different colour in terms of RGB [0, 0, 0] to [255, 255, 255]
    Custom version of skimage.color.label2rgb which uses int data type to save memory
    """
    # cols=np.random.randint(0, 255, (np.max(labels)+10, 3), dtype='uint8'); cols[0]=[0,0,0]; cols[1]=[255,255,255]
    # cols=np.full((np.max(labels)+10, 3), 255, dtype='uint8'); cols[0]=[0,0,0];
    cols=np.full(((np.max(labels)+10, 3)), 255)
    for i in fib_group:
        cols[i]=np.random.randint(0, 255, (1, 3), dtype='uint8')
    cols[0]=[0,0,0];cols[1]=[255,255,255]
    RGB_vol=np.zeros((labels.shape[0], labels.shape[1], labels.shape[2], 3), dtype='uint8')
    for i in np.arange(labels.shape[0]):
        RGB_plane=cols[(labels[i]+1)] #https://forum.image.sc/t/skimage-color-label2rgb-but-choose-specific-colors-for-specific-labels/62500
        RGB_vol[i]=RGB_plane
    return RGB_vol
def create_animation(fib_group, labels, startplane, endplane, dt, fig_size=10, step=1,colourful=True):
    """
    Create mapping animation object
    ORIGINAL THIS ONE WORKS
    """
    nplanes=labels.shape[0]
    if endplane==0:
        endplane=nplanes
    fig, ax=plt.subplots(1, 1, figsize=(fig_size,fig_size))
    container = [];    print("Making Label2RGB")
    rgblabel=custom_RGB_maker(fib_group,labels[startplane:endplane:step, :, :])
    frameID=0 ;     print("Creating animation")
    for pID in tqdm(range(startplane, endplane, step)):
        im=ax.imshow(rgblabel[frameID], animated=True)
        plot_title = ax.text(0.5,1.05,'Plane %d of %d' % (pID, nplanes),
                 size=plt.rcParams["axes.titlesize"],
                 ha="center", transform=ax.transAxes, )
        container.append([im, plot_title])
        frameID+=1;
    ani = animation.ArtistAnimation(fig, container, interval=dt, blit=True)
    plt.close();
    return ani

def export_animation(resultsDir,fib_group, labels, title='volumerendering/stepthrough-animation', startplane=0, endplane=0, dt=500, figsize=20,step=1):
    ani=create_animation(fib_group, labels, startplane, endplane, dt, figsize , step)
    ani.save(resultsDir+title+'.mp4')
def animation_inline(fib_group, labels, startplane=0, endplane=0, dt=500, figsize=10, step=1, fas_coords=0):
    """
    Animate a certain group of fibrils, given a specific fibril record
    """
    ani=create_animation(fib_group, labels, startplane, endplane, dt, figsize, step, fas_coords=fas_coords)
    return HTML(ani.to_html5_video())


#---------------PLOTS-------------------
def my_histogram(arr,xlabel, show, dens=False, title=0, labels=[], binwidth=10, xlims=0, pi=False,filename=0, leg=False, fitdata=0, fitparams=0, units=''):
    """
    A histogram, with number on the y axis
    """
    #cols=['red', 'lime', 'blue', 'pink']
    fig, ax=plt.subplots()
    minx=np.min(np.concatenate(arr)) if isinstance(arr,list) else np.min(arr)
    maxx=np.max(np.concatenate(arr)) if isinstance(arr,list) else np.max(arr)
    bins=np.arange(binwidth * np.floor(minx/binwidth),binwidth * (1+np.ceil(maxx/binwidth)),binwidth)
    ax.hist(arr, bins=bins, density=dens, histtype='bar', edgecolor='black', label=labels)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    ylabel='Probability density' if dens else 'Number'
    ax.set_ylabel(ylabel); ax.margins(0);
    if xlims!=0:
        ax.xlim(xlims)
    if title !=0:
        ax.set_title(title)
    if pi:
        ax.set_xticks(np.arange(-np.pi/4, np.pi/2, step=(np.pi/4)), ['-π/4','0','π/4'])

    if leg:
        ax.legend()


    if np.any(fitparams):
        if len(fitparams)<=2:
            textstr=f'$\mu$ = {fitparams[1]:.3f}{units}\n$\sigma$ = {fitparams[0]:.3f}{units}'
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place a text box in upper left in axes coords
            ax.text(0.8, 0.9, textstr, transform=ax.transAxes, fontsize=18,
            verticalalignment='top', bbox=props)
        else:
            ax.margins(0.05, 0.05)
            t0=f'$w={fitparams[4]:.3f}$\n'
            t1=f'$\mu_1$ = {fitparams[1]:.0f} nm, $\sigma_1$ = {fitparams[0]:.1f} nm, $w={fitparams[4]:.2f}$\n'
            t2=f'$\mu_2$ = {fitparams[3]:.0f} nm, $\sigma_2$ = {fitparams[2]:.1f} nm'
            textstr=t1+t2
            # these are matplotlib.patch.Patch properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # place a text box in upper left in axes coords
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    if np.any(fitdata):
        if dens:
            ax.plot(fitdata[0], fitdata[1], 'r')
        else:
            ax2=ax.twinx()
            ax2.plot(fitdata[0], fitdata[1], 'r')

            ax2.set_ylabel('Probability density')
            ax2.margins(0)
            ax2.grid(False)
    if filename!=0:
        plt.savefig(filename)
    if show:
        plt.show()

#------------------------------STATS---------------------

def moving_avg(data, window_width):
    """
    A rolling average over a 1d array
    """
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec
