from a0_initialise import *
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
plt.style.use('./mystyle.mplstyle')

print("b2: Volume rendering")
def load_FTA_results_rough():
    md.create_Directory(d.dirOutputs+'/volumerendering')
    try:
        path=glob.glob( d.dirOutputs + 'morphComp*')[0];
        MC=np.load(path) #original, import fibril record
    except:
        print("Error, no MC found")
    try:
        path=glob.glob( d.dirOutputs + 'fib_rec*')[0];
        FR=np.load(path) #original, import fibril record
    except:
        FR=0
        print("Error, no fibrec found")
    nF=FR.shape[0];print(f'nF={nF}')
    try:
        volume=md.label_volume(MC,np.arange(nF),FR,d.nP)
        if np.any(d.junk <=d.nP): #CORRECTING JUNK PLANES
            volume=np.delete(volume,d.junk[d.junk<=d.nP], axis=0) #deletes junk planes
        # print (d.dirOutputs+f'labelled_vol_{d.frac:.2f}')
        np.save(d.dirOutputs+f'labelled_vol_{100*d.frac}', volume)
    except:
        volume=0
        print("Error, no labelled volume found")

    return MC, FR, volume, nF
# MC, FR, volume,d.frac=load_FTA_results_rough()

def load_FTA_results():
    md.create_Directory(d.dirOutputs+'/volumerendering')
    try:
        path=glob.glob( d.dirOutputs + 'morphComp*')[0];
        MC=np.load(path) #original, import fibril record
    except:
        print("Error, no MC found")
    try:
        path=glob.glob( d.dirOutputs + 'fib_rec*'+f'{d.frac*100}*')[0];
        FR=np.load(path) #original, import fibril record
        nF=FR.shape[0]
        print(f'nF={nF}')
    except:
        FR=0
        print("Error, no fibrec found")
    try:
        path=glob.glob( d.dirOutputs + f'label*{d.frac*100}*')[0]
        volume=np.load(path) #original, import fibril record
        # if np.any(junk <=d.nP): #CORRECTING JUNK PLANES
        #     volume=np.delete(volume,junk[junk<=d.nP], axis=0) #deletes junk planes
    except:
        volume=0
        print("Error, no labelled volume found")
    return MC, FR, volume, nF
# MC, FR, volume,d.frac=load_FTA_results()
#%%----------------STEP THROUGH ANIMATION ---------------------------------

def stepthrough():
    md.export_animation(d.dirOutputs, np.arange(nF) ,volume,dt=100,step=1)

#%%----------------MID -PLANE IMAGE------------------------------------------
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
    plt.imshow(label2rgb(volume[int(d.nP/2)], bg_label=-1,colors=color), extent=[0,d.npix*d.pxsize/1000,0, d.npix*d.pxsize/1000])
    plt.savefig(d.dirOutputs+'volumerendering/mid-stack-img');
    plt.show()
# midPlaneImage()

#%%------------------VOLUME RENDERING------------------------------------------

def volume_render(labels, z1, z2, x1, x2,  pxsize,dz,dirOutputs,filename, el=40,aspect=True, show=True):
    minivol=labels[z1:z2, x1:x2, x1:x2]#subsection
    # minivol=volume
    #plotting
    fig = plt.figure(figsize=(20, 15))
    ax=plt.axes(projection='3d')
    whichfibs=np.unique(minivol)[np.unique(minivol)>0]
    # print(whichfibs)
    j=0
    print("Volume rendering image")
    for i in tqdm(whichfibs):
        j+=1
        print(f"fibril {j} of {len(whichfibs)}") if j in np.arange(0, len(whichfibs), len(whichfibs//10)) else 0
        minivol_coords=np.argwhere(minivol==i)
        # print(minivol_coords)
        ax.scatter(d.pxsize*minivol_coords[:,1]/1000, d.pxsize*minivol_coords[:,2]/1000, dz*minivol_coords[:,0]/1000, marker=',')

    proportions=(d.pxsize*minivol.shape[1],d.pxsize*minivol.shape[2],dz*minivol.shape[0])
    if aspect:
        ax.set_box_aspect(proportions)  # aspect ratio is 1:1:1 in data space

    ax.view_init(elev=el, azim=225)

    ax.set_xlabel('x ($\mu$m)');ax.set_zlabel('z ($\mu$m)');ax.set_ylabel('y ($\mu$m)')
    print("Saving VR rendering image")
    plt.savefig(d.dirOutputs+filename+'.png')
    if show:
        plt.show()



#%%
# d.dirOutputs
# MC, FR, volume,nF=load_FTA_results()
# labels=volume
# fib_group=np.arange(nF)
#
#
#
# cols=np.random.randint(0, 255, (len(fib_group), 3), dtype='uint8'); cols[0]=[0,0,0]
# RGB_vol=np.zeros((labels.shape[0], labels.shape[1], labels.shape[2], 3), dtype='uint8')
#
# for i in np.arange(labels.shape[0]-1):
#     RGB_plane=cols[(volume[i]+1)] #https://forum.image.sc/t/skimage-color-label2rgb-but-choose-specific-colors-for-specific-labels/62500
#     RGB_vol[i]=RGB_plane
#
#

#%%-----------MAIN FLOW

if __name__=='__main__':
    # parallell_process()
    MC, FR, volume,nF=load_FTA_results()
    stepthrough()
    midPlaneImage()
    volume_render(volume, 0, d.nP, 0, d.npix, d.pxsize, d.dz, d.dirOutputs, 'volumerendering/volume-render')


#%%------------------VOLUME FOR ABC-------------------------------------------------

# abc_sorted=np.loadtxt('/Users/user/Dropbox (The University of Manchester)/fibril-tracking/nuts-and-bolts/csf-695/abc/top5/a1_b_c_values_sorted.txt')

def find_fibrecs():
    top5_fs=glob.glob('/Users/user/Dropbox (The University of Manchester)/fibril-tracking/nuts-and-bolts/csf-695/abc/*/*/fibrec*');top5_fs.sort()
    lis=glob.glob('/Users/user/Dropbox (The University of Manchester)/fibril-tracking/nuts-and-bolts/csf-695/abc/top3-sensible-and-bottom3-16Dec/*.npy');lis.sort()
    bottom3_fs=lis[0:3] ; top_3_reasonable=lis[3:]
    orig111_f=['/Users/user/Dropbox (The University of Manchester)/fibril-tracking/nuts-and-bolts/csf-695/abc/fib_rec_a_1.00_b_1.00_c_1.00.npy']
    return top5_fs,bottom3_fs, top_3_reasonable, orig111_f

def get_abc_from_filename(f):
    a=float(f.split('_' )[-5])
    b=float(f.split('_' )[-3])
    c=float(f.split('_', )[-1][0:4])
    return a,b,c

def func(f, output_prefix='test', seed=0):
    MC, _ , _, d.pxsize,junk, dz, d.nP, d.npix, d.frac=load_FTA_results()
    print (f"ABC={get_abc_from_filename(f)}")
    FR=np.load(f)
    print(f'nF before trim {FR.shape}')
    FR_trim=trim_fib_rec(FR, MC, frac=d.frac)
    nF=FR_trim.shape[0]
    print(f'nF after trim {nF}')
    labels=label_volume(MC,np.arange(0, nF, nF//50),FR_trim, d.nP)
    d.dirOutputs='/Users/user/Dropbox (The University of Manchester)/fibril-tracking/nuts-and-bolts/csf-695/abc/'
    a,b,c=get_abc_from_filename(f)
    outputfilename=f'{output_prefix}_a_{a}_b_{b}_c_{c}'
    volume_render(labels, 0, d.nP, 0, 1000, d.pxsize,dz, d.dirOutputs, outputfilename, show=False)

def parallell_process():
    top5_fs,bottom3_fs, top_3_reasonable, orig111_f=find_fibrecs()
    start=time.perf_counter()
    li=[]
    for i in range(5):
        t=multiprocessing.Process(target=func, args=(top5_fs[i], f'/top5/top5_{i}', i))
        t.start();li.append(t)
    for t in li:
        t.join()

    li2=[]
    for i in range(3):
        t2=multiprocessing.Process(target=func, args=(bottom3_fs[i], f'/top3-sensible-and-bottom3-16Dec/bottom_3_{i}', i+10))
        t2.start();li2.append(t2)
    for t2 in li2:
        t2.join()
    li3=[]
    for i in range(3):
        t3=multiprocessing.Process(target=func, args=(top_3_reasonable[i], f'/top3-sensible-and-bottom3-16Dec/top3_sensible_{i}',i+20))
        t3.start();li3.append(t3)
    for t3 in li3:
        t3.join()
    func(orig111_f[0], 'original')
    finish=time.perf_counter()
    print (f'Finished in {round((finish-start)/60, 5)}s')

#%%---------------VOLUME RENDERING ANIMATION SANDBOX-----------------------------
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
    ax.scatter(d.pxsize*minivol_coords[:,1], d.pxsize*minivol_coords[:,2], dz*minivol_coords[:,0], marker=',')
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
for pID in range(1, d.nP):

    print(f'animating {pID}')
    minivol=volume[0:pID, 0:300, 0:300]
    for i in whichfibs:
        minivol_coords=np.argwhere(minivol==i)
        im=ax.scatter(d.pxsize*minivol_coords[:,1], d.pxsize*minivol_coords[:,2], dz*minivol_coords[:,0], marker=',')
    ax.set_box_aspect(proportions)  # aspect ratio is 1:1:1 in data space
    ax.view_init(elev=30, azim=225)
    ax.set_xlabel('x (nm)');ax.set_zlabel('z ($\mu$m)');ax.set_ylabel('y ($\mu$m)')
    proportions=(d.pxsize*minivol.shape[1],d.pxsize*minivol.shape[2],dz*minivol.shape[0])
    ax.set_box_aspect(proportions)  # aspect ratio is 1:1:1 in data space
    ax.view_init(elev=30, azim=225)
    ax.set_xlabel('x (nm)');ax.set_zlabel('z ($\mu$m)');ax.set_ylabel('y ($\mu$m)')
    container.append([im])

ani = animation.ArtistAnimation(fig, container, interval=1000, blit=True)
plt.close();
HTML(ani.to_html5_video())
"""

def f():
    #TRYING TO MAKE OWN RGB LABEL
    import importlib
    importlib.reload(md);

    # labels=md.label_volume(MC, np.arange(nF), FR, 110, 100)
    # junk
    # md.animation_inline(np.arange(nF) ,labels, step=1)


    labels=md.label_volume(MC, np.arange(nF), FR, 10)

    nF=np.max(labels)+1
    fib_group=np.unique(labels)
    fib_group
    labels.shape
    color = [tuple(np.random.random(size=3).astype('float16')) for i in range(nF)]
    color.insert(0, (1.,1.,1.))
    pID=0
    labels.shape[1]

    rgblabel=np.zeros((labels.shape[1],labels.shape[2], 3))
    for i in fib_group:
        #includes -1
        rgblabel[np.argwhere(labels[pID]==0)]=color[i]

    plt.imshow(rgblabel)

    labels[pID]
