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


def printarray(array):
    """
    Print an aligned array
    """
    print(pd.DataFrame(array).to_string(index=False, header=False))

def showmorphComp(i, morphComp):
    """
    Displays a colourised plane, of all the morphological components
    """
    image_label_overlay = label2rgb(morphComp[i], bg_label=0)
    fig, ax = plt.subplots()
    ax.imshow(image_label_overlay, interpolation='nearest')
    plt.title('Plane %i of %i' % (i, morphComp.shape[0]))
    plt.ylabel('y pix')
    plt.xlabel('x pix')
    plt.show()

def t_d_stamp():
    return "{:%y-%m-%d_%H-%M}".format(dt.now())
def t_stamp():
    return "{:%H-%M-%S}".format(dt.now())


def beep():
    winsound.Beep(500, 800);

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


def label_volume(morphComp,fib_group,fib_rec,endplane, startplane=0):
 labels=np.where(morphComp==0, -1, 0) #turn all fibs to 0, keeping background=-1
 for pID in range(startplane, endplane):
     for i in range (len(fib_group)):
         if fib_rec[fib_group[i], pID]!=-1:
             value=fib_group[i]+1;
             labels[pID]=np.where(morphComp[pID]==fib_rec[fib_group[i], pID]+1, value, labels[pID])
 return labels;


def export_mapping_animation(exportDir,morphComp,fib_group, fib_rec,title, endplane, startplane=0 ):
 """
 Export mapping animation
 """
 fig = plt.figure()
 ims=[];
 labels=label_volume(morphComp,fib_group, fib_rec, endplane,startplane);
 color = [tuple(np.random.random(size=3)) for i in range(5000)] #randomcolourlist
 color.insert(0,(1.,1.,1.)) #makesure other fibrils are white!!
 rgblabel=label2rgb(labels, bg_label=-1, colors=color);
 for pID in range(startplane, endplane):
     im=plt.imshow(rgblabel[pID], animated=True)
     ims.append([im])
 ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
 plt.close();
 ani.save(exportDir+title+'.mp4')


def view_fibs_inline_2(morphComp,fib_group, fib_rec, endplane, startplane=0,save=False, axis=3):
    """
    Animate a certain group of fibrils, given a specific fibril record
    """
    fig = plt.figure();

    ims=[];
    labels=label_volume(morphComp,fib_group, fib_rec, endplane, startplane);
    color = [tuple(np.random.random(size=3)) for i in range(5000)] #randomcolourlist
    color.insert(0,(1.,1.,1.)) #makesure other fibrils are white!!
    rgblabel=label2rgb(labels, bg_label=-1, colors=color);

    dt=500;
    if axis==1:
        for x in range (npix):
            ims.append([plt.imshow(rgblabel[:, x, :],aspect=dz/pxsize)])
            #might need ficxing
    elif axis==2:
        for y in range (npix):
            ims.append([plt.imshow(rgblabel[:, :, y],aspect=dz/pxsize)])
    else:
       for pID in range(startplane, endplane):
        im=plt.imshow(rgblabel[pID], animated=True)
        ims.append([im])


    anim = animation.ArtistAnimation(fig, ims, interval=dt, blit=True, repeat_delay=1000)
    plt.close(); #suppress blank plot
    #return labels

    return HTML(anim.to_html5_video())
