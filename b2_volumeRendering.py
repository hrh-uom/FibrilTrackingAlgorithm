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
import a0_initialise as a0

plt.style.use('./mystyle.mplstyle')

print(f'b2: Volume rendering {md.t_d_stamp()}')

def load_FTA_results():
    d, MC, props = a0.initialise_dataset()
    md.create_Directory(d.dirOutputs+'/volumerendering')
    try:
        # path=glob.glob( d.dirOutputs + 'fib_rec*'+f'{d.frac*100}*')[0];
        path=glob.glob( d.dirOutputs + 'fib_rec_trim*')[0]
        FR=np.load(path) #original, import fibril record
        nF=FR.shape[0]
        print(f'nF={nF}')
    except:
        FR=0
        print("Error, no fibrec found")
    try:
        path=glob.glob( d.dirOutputs + f'label*')[0]
        volume=np.load(path) #original, import fibril record
        # if np.any(junk <=d.nP): #CORRECTING JUNK PLANES
        #     volume=np.delete(volume,junk[junk<=d.nP], axis=0) #deletes junk planes
    except:
        volume=0
        print("Error, no labelled volume found")
    return d, MC, props, FR, volume, nF

#----------------STEP THROUGH ANIMATION ---------------------------------

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
    # plt.show()
# midPlaneImage()

#------------------VOLUME RENDERING------------------------------------------
def volume_render(labels, d, z1, z2, x1, x2,filename,resamplex=1, resamplez=1,el=40,aspect=True, show=True):
    minivol=labels[z1:z2:resamplez, x1:x2:resamplex, x1:x2:resamplex]#resampled volume
    print("Volume rendering image")
    fig = plt.figure(figsize=(30, 20)) ; ax = fig.add_subplot(projection='3d')
    whichfibs=np.unique(minivol)[np.unique(minivol)>0] ; j=0
    for i in tqdm(whichfibs):
        minivol_coords=np.argwhere(minivol==i)
        ax.scatter(resamplex*d.pxsize*minivol_coords[:,1]/1000, resamplex*d.pxsize*minivol_coords[:,2]/1000, resamplez*d.dz*minivol_coords[:,0]/1000, marker=',')
    # ax.scatter(np.random.randint(0, 10000, 100)/1000, np.random.randint(0, 10000, 100)/1000, np.random.randint(0, 50000, 100)/1000)
    proportions=(resamplex*d.pxsize*minivol.shape[1],resamplex*d.pxsize*minivol.shape[2],resamplez*d.dz*minivol.shape[0])

    print(f'proporions {d.pxsize*d.npix}, {d.dz*d.nP_all}')
    ax.set_box_aspect(proportions)  # aspect ratio is 1:1:1 in data space
    ax.view_init(elev=el, azim=225)

    #LABELS
    label_sz=24
    ax.set_xlabel('x ($\mu$m)', labelpad=15, fontsize=label_sz);
    ax.set_ylabel('y ($\mu$m)', labelpad=15, fontsize=label_sz);
    ax.set_zlabel('z ($\mu$m)', rotation='horizontal', labelpad=90 , fontsize=label_sz, )

    #TICKS
    ticksize=18;
    ax.tick_params(axis='x', pad=0, labelsize=ticksize);
    ax.tick_params(axis='y', pad=-5, labelsize=ticksize, rotation = 45);
    ax.tick_params(axis='z', pad=30, labelsize=ticksize)

    print("Saving VR rendering image");
    plt.savefig(d.dirOutputs+filename+f'_resample_{resamplex}_{resamplez}'+'.png')
    if show:
        plt.show()

#%%-----------MAIN FLOW
if __name__=='__main__':
    d, MC, props, FR, volume, nF=load_FTA_results()
    stepthrough()
    midPlaneImage()
    volume_render(volume, d, 0, d.nP, 0, d.npix, 'volumerendering/volume-render', resamplex=2, resamplez=2, show=False)

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
