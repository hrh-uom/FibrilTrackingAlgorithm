import numpy as np
from numpy.random import randint
import pandas as pd
from matplotlib import animation, rc, colors
import matplotlib.pyplot as plt
from PIL import Image
import glob
from IPython.display import HTML
from skimage.measure import label, regionprops,regionprops_table
from skimage.color import label2rgb
from time import time as time_s
import winsound #for development
import importlib
import fibtrack_custom_functions as md
import os
import csv
plt.rcParams['figure.figsize'] = [10, 7.5];#default plot size
plt.rcParams['font.size']=16;
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['animation.ffmpeg_path'] = 'C:\\FFmpeg\\bin\\ffmpeg.exe'; # SPECIFIC TO YOUR MACHINE, for inline animations

#----------------------------------------------------------------------------------
#.........................1. IMPORT DATA AND IMAGES.....................................
#------------------------------------------------------------------------------------

def create_Directory(directory):
    """
    Tests for existence of directory then creates one if not
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
def M1_import_images(whichdata=15):
    """
    WHICHDATA=CHOOSE TIMEPOINT IN 24H CLOCK
    """
    global mainDir
    global imgstack
    if whichdata==9:
        mainDir = 'D:\\3View\\9am-achilles-fshx\\7.5um_crop_2\\';
    elif whichdata==19:
        mainDir = 'D:\\3View\\7pmAchx700\\7-5um\\';
    elif whichdata==19.100: #dummyset1!!
        mainDir='D:\\3View\\7pmAchx700\\7-5um\\dummy_0_100\\'
    elif whichdata==9.15: #dummyset1!!
        mainDir='D:\\3View\\9am-achilles-fshx\\7.5um_crop_2\\dummy_110_124\\'
    elif whichdata==9.100:
        mainDir='D:\\3View\\9am-achilles-fshx\\7.5um_crop_2\\dummy_100_200\\'
    else:
        return 0;
    imagePath = glob.glob(mainDir + 'fin\\*.tif');
    imgstack_neg = np.array( [np.array(Image.open(img).convert('L'), 'uint8') for img in imagePath]);
    imgstack=np.logical_not(imgstack_neg).astype(int); #May not always be necessary to invert!
def M2_junk_slices_and_dimensions(skip=1):
    global junk
    global pxsize
    global dz
    global nplanes
    global npix
    global Lscale
    global exportDir
    global imgstack
    global mainDir
    pxsize, dz=np.genfromtxt(mainDir+'pxsize.csv', delimiter=',')[1];
    nplanes, npix, npix=imgstack.shape
    Lscale = 200/pxsize;
    if skip>1: #Resize array and renumber junk slices if skipping slices
        keep=skip*np.arange(nplanes/skip).astype(int)
        imgstack=imgstack[keep];
        junk=pd.read_csv(mainDir+'junkslices.csv', header=None).to_numpy().T[0]/skip;
        dz*=skip;
        nplanes, npix, npix=imgstack.shape
        exportDir=mainDir+'skip_%d_results\\'%skip;
    else:
        #Make a directory which stores results
        junk=pd.read_csv(mainDir+'junkslices.csv', header=None).to_numpy().T[0];
        exportDir=mainDir+'results\\';
    [create_Directory(i) for i in [mainDir,exportDir]];

#----------------------------------------------------------------------------------
#.........................2. LABEL AND MEASURE .....................................
#------------------------------------------------------------------------------------
def M3_label_and_measure():
    global morphComp
    global props
    #Extracting morphological components and properties of interest ------------------
    morphComp=np.empty(imgstack.shape, dtype=int) #initialising array for morph comp
    for i in range(nplanes):
        morphComp[i]=label(imgstack[i])

    #Show particular image plane with labeled components
    #md.showmorphComp(randint(0,nplanes), nplanes, morphComp)

    """
    Setting up table of properties for each plane (props)
    props stores (pID, objectID, property). the +1 is because centroid splits into 2. it is the length of the max number of objects in any plane, and populated with zeroes.
    """
    props_ofI='centroid','orientation','area','eccentricity','minor_axis_length';
    props=np.empty((nplanes, np.max(morphComp), len(props_ofI)+1));
    for pID in range(nplanes):
        temp=pd.DataFrame(regionprops_table(morphComp[pID], properties=props_ofI)).to_numpy();
        props[pID,0:temp.shape[0]]=temp;

    np.save(exportDir+'props', props);
    np.save(exportDir+'morphComp', morphComp);
#---------------------------------------------------------------------------
#.................................3. FIBRIL MAPPING...........................
#-------------------------------------------------------------------------------
#------Errors
def err_c_v(pID, i, j, dz_f):
    """
    centroid error without prior knowledge of trajectory (vertical mapping)
    """
    return np.linalg.norm(props[pID, i, 0:2]-props[pID+dz_f, j, 0:2])/Lscale
def err_c(pID, i,prev_i, j, dz_b, dz_f):
    """
    centroid error WITH prior knowledge of trajectory (differential mapping)
    """
    currentcent=props[pID, i, 0:2];
    prevcent=props[pID-dz_b, prev_i, 0:2];
    predictedcent=currentcent+dz_f*(currentcent-prevcent); #add a dz here
    return np.linalg.norm(predictedcent-props[pID+dz_f, j, 0:2])/Lscale
def err_e(pID, i, j, dz_f):#error in eccentricity
 return  np.abs(props[pID, i, 4]-props[pID+dz_f, j, 4])
def err_a(pID, i, j, dz_f): #error in area
 return np.abs(props[pID+dz_f, j, 3]-props[pID, i, 3])/props[pID, i, 3];
def err(pID, fID, prev_i, j,dz_b, dz_f, a, b, c):  #not ensuring values need to ve <1
    if pID==0:
        return (1/(a+b+c)) *(a*err_c_v(pID, fID, j,dz_f)+b*err_e(pID, fID, j,dz_f)+c*err_a(pID, fID, j,dz_f))
    else:
        return (1/(a+b+c)) *(a* err_c(pID, fID, prev_i,j,dz_b, dz_f)+b*err_e(pID, fID, j,dz_f)+c*err_a(pID, fID, j,dz_f));
def errorthresh(a, b, c, skip): #max values for error cutoff.
 return (1/(a+b+c))*(a * Lscale/(skip*Lscale) + b * 0.2 + c *0.5) #equiv to 300nm centroid jump, de=0.2 and dA=50% inc in size
#--------Junk Slice Functions
def increments_back_forward(pID, junk):
    """
    skipping slices in the case of junk slices
    """
    test_plane=pID;inc_fw=1;
    while np.any(junk==test_plane+1):
     inc_fw=inc_fw+1;
     test_plane=test_plane+1;
    test_plane=pID;inc_back=1;
    while np.any(junk==test_plane-1):
     inc_back=inc_back+1;
     test_plane=test_plane-1;
    return inc_back,inc_fw;
def lastplane_tomap(junk):
    """
    finding the penultimate plane, incase the last / penultimate is a junk slice
    """
    pID=nplanes;
    while np.any(junk==pID):
     pID-=1;
    return pID-increments_back_forward(pID, junk)[0];
def M4_fibril_mapping(a,b,c, skip):
    global nfibs
    global fib_rec
    global nplanes
    start_time=time_s();
    f = open(exportDir+r'\fibtrack_status_update.csv', "a")
    f.write('\ntime '+md.t_d_stamp()+'\nJunk slices,'+str(junk)+"\npID,nfibs,time since mapping began");
    f.close()
    nfibs=np.max(morphComp[0]);#number eqivalent to n objects in first slice
    fib_rec=np.full((nfibs,nplanes),-1, dtype=int); #-1 means no fibril here, as indices are >= 0
    fib_rec[:,0]=np.arange(nfibs); #use like fib_rec[fID, pID]
    for pID in range (lastplane_tomap(junk)):
        if np.any(junk==pID):#If the slice is junk, skip the mapping.
            #x=1;
            continue;
        dz_b, dz_f=increments_back_forward(pID,junk)
        err_table=np.zeros([nfibs,np.max(morphComp[pID+dz_f])]); #table of errors i,j.Overwritten at each new pID

        #CREATING ERROR TABLES
        for fID in range(nfibs):
            #Isolating the relevant 'patch in morphological components
            if fib_rec[fID,pID]!=-1: # catching nonexistent fibrils, true in pID>0
                cofI=props[pID,fib_rec[fID,pID],0:2]#centroid of fibril in plane
                index=np.ndarray.flatten(md.search_window(cofI, npix/10, npix)).astype('int')
                compare_me=np.delete(np.unique(np.ndarray.flatten(morphComp[pID+dz_f,index[0]:index[1], index[2]:index[3]]-1) ),0);#find a more neat way to do this. List of indices in next slice to look at.
                for j in compare_me: #going through relevant segments in next slice
                    err_table[fID,j]=err(pID, fib_rec[fID,pID], fib_rec[fID,pID-dz_b], j,dz_b, dz_f, a, b, c);

        #sorted lists of the errors and the pairs of i,j which yield these Errors
        sort_errs=sort_errs=np.sort(err_table, axis=None)
        sort_err_pairs =np.transpose(np.unravel_index(np.argsort(err_table, axis=None), err_table.shape));

        #delete pairs with 0 errors (ie those who are outside the box) and those above the threshold
        delete=np.concatenate((np.where(sort_errs==0)[0],np.where(sort_errs>errorthresh(a,b,c, skip))[0]), axis=0)
        sort_err_pairs=np.delete(sort_err_pairs,delete, axis=0)

        i=0; #Matching up
        while sort_err_pairs.shape[0]>0:
            match=sort_err_pairs[0]; # picks out smallest error match
            fib_rec[match[0], pID+dz_f]=match[1]; # fills in the corresponding fibril recor with this match
            #delete all other occurences of i,j
            deleteme=np.unique(np.ndarray.flatten(np.concatenate((np.argwhere(sort_err_pairs[:,0]==match[0]),np.argwhere(sort_err_pairs[:,1]==match[1]))))).tolist()
            sort_err_pairs=np.delete(sort_err_pairs, deleteme,axis=0)
            i=i+1;

        #LOOK FOR MISSED FIBRILS
        all=np.arange(np.max(morphComp[pID+dz_f]));
        mapped=np.unique(fib_rec[:,pID+dz_f][fib_rec[:,pID+dz_f]>-1]);
        new_objects=np.setdiff1d(all, mapped);
        fibrec_append=np.full((new_objects.size,nplanes),-1, dtype=int); #an extra bit to tack on the end of the fibril record accounting for all these new objects
        fibrec_append[:,pID+dz_f]=new_objects;
        fib_rec=np.concatenate((fib_rec,fibrec_append));
        nfibs+=new_objects.size;

        # save/export stuff
        f = open(exportDir+r'\fibtrack_status_update.csv', "a")
        f.write('\n'+','.join(map(str,[pID,nfibs,time_s()-start_time])))
        f.close()
        np.save(exportDir+'fib_rec', fib_rec);
#---------------------------------------------------------------------------
#..................................4.ANIMATION/VISUALISATION.................
#-------------------------------------------------------------------------------
def ani_fib(fID, fib_rec):
 fig = plt.figure();ims=[];
 for pID in range(4):
     labelled=np.where((morphComp[pID]!=0)&(morphComp[pID]!=fib_rec[fID,pID]+1), 1000, morphComp[pID]);
     rgblabel=label2rgb(labelled, bg_label=0, colors=[(1, 0, 0), (1, 1, 1)]);
     ims.append([plt.imshow(rgblabel)])
 plt.close(); #suppress blank plot
 anim = animation.ArtistAnimation(fig, ims, interval=500);
 return HTML(anim.to_html5_video());

def red_objects_1_plane(obj_group, pID):
 """
 Please feed object numbers (inplane) not fibril numbers
 """
 labels=morphComp[pID].copy();
 for i in range(obj_group.size):
     value=obj_group[i];
     labels=np.where(morphComp[pID]==value+1, -1, labels)
 labels=np.where(labels>0, 1, labels)
 rgblabel=label2rgb(labels, bg_label=0, colors=[(1, 0, 0), (1, 1, 1)]);
 plt.imshow(rgblabel)
def timetrial():
    times=[whichdata,skip,time_s()-start_time]
    skip_filename=r'C:\Users\t97721hr\Dropbox (The University of Manchester)\Fibril Tracking Algorithm\skipping_timetrials.csv';
    with open(skip_filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(times)

#---------------------------------------------------------------------------
#..........................5. MAIN FLOW OF ALGORITHM...........................
#-------------------------------------------------------------------------------
start_time=time_s();

whichdata=9.15 ;skip=1;
a, b, c=10,0.1,5;

M1_import_images(whichdata);
M2_junk_slices_and_dimensions(skip);
M3_label_and_measure()

#%%
M4_fibril_mapping(a, b, c, skip);




#%%SANDBOX



view_fibs_inline_2(morphComp,np.arange(nfibs), fib_rec, endplane=nplanes, startplane=0,save=False, axis=3) # animate a certain group of fibrils
#
#%%%
plt.imshow(label2rgb(morphComp[randint(15)], bg_label=0))
morphComp.shape[0]
