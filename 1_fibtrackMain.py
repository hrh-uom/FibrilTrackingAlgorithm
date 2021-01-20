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
from importlib import reload # reload(module_of_choice);
import customFunctions as md
import os
import csv
from feret_diameter import feret_diameters_2d
plt.rcParams['figure.figsize'] = [10, 7.5] #default plot size
plt.rcParams['font.size']=16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['animation.ffmpeg_path'] = 'C:\\FFmpeg\\bin\\ffmpeg.exe'  # SPECIFIC TO YOUR MACHINE, for inline animations

#----------------------------------------------------------------------------------
#...............................1. USER INPUT .................................
#------------------------------------------------------------------------------------
fromscratch=False
whichdata=0;skip=1
#a, b, c=10,0.1,5

#----------------------------------------------------------------------------------
#.........................2. IMPORT IMAGES , LABEL, MEASURE ......................
#------------------------------------------------------------------------------------
def create_binary_stack(dir3V):
    """
    imports images from given 3V directory
    """
    imagePath = glob.glob( dir3V + 'fin\\*.tif')
    imgstack_neg = np.array( [np.array(Image.open(img).convert('L'), 'uint8') for img in imagePath])
    imgstack=np.logical_not(imgstack_neg).astype(int)  #May not always be necessary to invert!
    return imgstack
def compress_by_skipping(skip):
    global dz
    global nplanes
    global junk
    global imgstack
    global dirResults
    if skip>1: #Resize array and renumber junk slices if skipping slices
        keep=skip*np.arange(nplanes/skip).astype(int)
        imgstack=imgstack[keep]
        junk=junk/skip
        dz*=skip
        nplanes=imgstack.shape[0]
        dirResults= dir3V+'skip_%d_results\\'%skip
def create_morph_comp(imgstack):
    global npix
    global nplanes
    morphComp=np.zeros([nplanes, npix, npix], dtype=int) #initialising array for morph comp
    for i in range(nplanes):
        morphComp[i]=label(imgstack[i])
    np.save(dirResults+'morphComp', morphComp)
    return morphComp
def create_properties_table(morphComp):
    """
    Setting up table of properties for each plane (props) props stores (pID, objectID, property). it is the length of the max number of objects in any plane, and populated with zeroes.
    """
    props_ofI='centroid','orientation','area','eccentricity' # these properties are the ones to be calculated using skimage.measure
    props=np.empty((nplanes, np.max(morphComp), len(props_ofI)+2)) #the +2 is because centroid splits into 2, and also to leave space for the feret diameter, calculated by a custom script.
    for pID in range(nplanes):
        rprops=pd.DataFrame(regionprops_table(morphComp[pID], properties=props_ofI)).to_numpy() #regionprops from skimage.measure
        #print (temp.shape)
        nobj=rprops.shape[0]; # nobjects in plane
        props[pID,0:nobj, 0:5]=rprops
        props[pID,0:nobj, 5]=feret_diameters_2d(morphComp[pID])
    np.save(dirResults+'props', props)
    return props
    #return temp.shape

dir3V=md.find_3V_data(whichdata); # find the relevant data based on the timepoint desired
pxsize, dz=np.genfromtxt( dir3V+'pxsize.csv', delimiter=',')[1] #import voxel size
junk=pd.read_csv( dir3V+'junkslices.csv', header=None).to_numpy().T[0] #which slices are broken
dirResults= dir3V+'results\\' #Create subfolder (if it doesnt already exist!)
md.create_Directory(dirResults)

if fromscratch:
    imgstack=create_binary_stack(dir3V) #import images and create binary array
    nplanes, npix, _ = imgstack.shape #measure array dims
    if skip>1:
        compress_by_skipping(skip)
    morphComp=create_morph_comp(imgstack);
    props=create_properties_table(morphComp)
else: #to save time
    morphComp=np.load(dirResults+"morphComp.npy")
    props=np.load(dirResults+"props.npy")
    nplanes, npix, _=morphComp.shape

Lscale=np.median(np.ravel(props[:,:,5])) # A typical lengthscale, calculated from the median value of feret diameter, in pixels. To find the equiv in nm, multiply by pxsize
#%%---------------------------------------------------------------------------
#.................................3. FIBRIL MAPPING...........................
#-------------------------------------------------------------------------------
#------Errors
def err_c(pID, i,prev_i, j, dz_b, dz_f):
    """
    centroid error
    """
    #TEMPORARY !!! VERTICAL MAPPING
    if pID!=-1: #top slice, no history.
        return np.linalg.norm(props[pID, i, 0:2]-props[pID+dz_f, j, 0:2])/Lscale
    else:
        currentcent=props[pID, i, 0:2]
        prevcent=props[pID-dz_b, prev_i, 0:2]
        predictedcent=currentcent+dz_f*(currentcent-prevcent)
        return np.linalg.norm(predictedcent-props[pID+dz_f, j, 0:2])/Lscale
def err_a(pID, i, j, dz_f): #error in area
 return np.abs(props[pID+dz_f, j, 3]-props[pID, i, 3])/props[pID, i, 3]

def err_f(pID, i, j, dz_f): #error in feret diameter
 return np.abs(props[pID+dz_f, j, 5]-props[pID, i, 5])/Lscale
def err(pID, fID, prev_i, j,dz_b, dz_f, a, b, c):  #not ensuring values need to ve <1
    return (1/(a+b+c)) *(a* err_c(pID, fID, prev_i,j,dz_b, dz_f)+b*err_f(pID, fID, j,dz_f)+c*err_a(pID, fID, j,dz_f))
def errorthresh(a, b, c, skip): #max values for error cutoff.
 return (1/(a+b+c))*(a *skip + b + c )
#--------Junk Slice Functions
def increments_back_forward(pID, junk):
    """
    skipping slices in the case of junk slices
    """
    test_plane=pID
    inc_fw=1
    while np.any(junk==test_plane+1):
     inc_fw=inc_fw+1
     test_plane=test_plane+1
    test_plane=pID
    inc_back=1
    while np.any(junk==test_plane-1):
     inc_back=inc_back+1
     test_plane=test_plane-1
    return inc_back,inc_fw
def lastplane_tomap(junk):
    """
    finding the penultimate plane, incase the last / penultimate is a junk slice
    """
    pID=nplanes
    while np.any(junk==pID):
     pID-=1
    return pID-increments_back_forward(pID, junk)[0]


def fibril_mapping(a,b,c,skip=1, fib_rec_filename="fib_rec"):
    global nfibs
    global fib_rec
    global nplanes
    start_time=time_s()
    with open(dirResults+r'\fibtrack_status_update.csv', 'a') as status_update:
        status_update.write("\n\n a %.2f, b %.2f, c %.2f"%(a, b, c)+'\ntime '+md.t_d_stamp()+'\nJunk slices,'+str(junk)+"\npID,nfibs,time since mapping began")
    nfibs=np.max(morphComp[0]) #number eqivalent to n objects in first slice
    fib_rec=np.full((nfibs,nplanes),-1, dtype=int)  #-1 means no fibril here, as indices are >= 0
    fib_rec[:,0]=np.arange(nfibs)  #use like fib_rec[fID, pID]

    for pID in range (lastplane_tomap(junk)):
        if np.any(junk==pID):#If the slice is junk, skip the mapping.
            #x=1
            continue
        dz_b, dz_f=increments_back_forward(pID,junk)
        err_table=np.zeros([nfibs,np.max(morphComp[pID+dz_f])])  #table of errors i,j.Overwritten at each new pID

        #CREATING ERROR TABLES
        for fID in range(nfibs):
            #Isolating the relevant 'patch in morphological components
            if fib_rec[fID,pID]!=-1: # catching nonexistent fibrils, true in pID>0
                cofI=props[pID,fib_rec[fID,pID],0:2]#centroid of fibril in plane
                index=np.ndarray.flatten(md.search_window(cofI, npix/10, npix)).astype('int')
                compare_me=np.delete(np.unique(np.ndarray.flatten(morphComp[pID+dz_f,index[0]:index[1], index[2]:index[3]]-1) ),0) #find a more neat way to do this. List of indices in next slice to look at.
                for j in compare_me: #going through relevant segments in next slice
                    err_table[fID,j]=err(pID, fib_rec[fID,pID], fib_rec[fID,pID-dz_b], j,dz_b, dz_f, a, b, c)

        #sorted lists of the errors and the pairs of i,j which yield these Errors
        sort_errs=sort_errs=np.sort(err_table, axis=None)
        sort_err_pairs =np.transpose(np.unravel_index(np.argsort(err_table, axis=None), err_table.shape))

        #delete pairs with 0 errors (ie those who are outside the box) and those above the threshold
        delete=np.concatenate((np.where(sort_errs==0)[0],np.where(sort_errs>errorthresh(a,b,c, skip))[0]), axis=0)
        sort_err_pairs=np.delete(sort_err_pairs,delete, axis=0)

        i=0  #Matching up
        while sort_err_pairs.shape[0]>0:
            match=sort_err_pairs[0]  # picks out smallest error match
            fib_rec[match[0], pID+dz_f]=match[1]  # fills in the corresponding fibril recor with this match
            #delete all other occurences of i,j
            deleteme=np.unique(np.ndarray.flatten(np.concatenate((np.argwhere(sort_err_pairs[:,0]==match[0]),np.argwhere(sort_err_pairs[:,1]==match[1]))))).tolist()
            sort_err_pairs=np.delete(sort_err_pairs, deleteme,axis=0)
            i=i+1

        #LOOK FOR MISSED FIBRILS
        all=np.arange(np.max(morphComp[pID+dz_f]))
        mapped=np.unique(fib_rec[:,pID+dz_f][fib_rec[:,pID+dz_f]>-1])
        new_objects=np.setdiff1d(all, mapped)
        fibrec_append=np.full((new_objects.size,nplanes),-1, dtype=int)  #an extra bit to tack on the end of the fibril record accounting for all these new objects
        fibrec_append[:,pID+dz_f]=new_objects
        fib_rec=np.concatenate((fib_rec,fibrec_append))
        nfibs+=new_objects.size

        # save/export stuff
        with open(dirResults+r'\fibtrack_status_update.csv', 'a') as status_update:
            status_update.write('\n'+','.join(map(str,[pID,nfibs,time_s()-start_time])))
        np.save(dirResults+fib_rec_filename, fib_rec)

    #print("N Tracked as a percentage: %.3f"%(np.count_nonzero(fib_rec>-1)/np.count_nonzero(fib_rec[:, p]>-1)))

#%%
def ndropped(a, b, c, pID_list):

    """
    figuring out how many dropped in a pair of planes
    """
    lis=[];
    for pID in pID_list:
        nfibs=np.max(morphComp[pID]);
        fib_rec=np.full((nfibs,nplanes),-1, dtype=int) #-1 means no fibril here, as indices are >= 0
        fib_rec[:,pID]=np.arange(nfibs); #use like fib_rec[fID, pID]
        dz_b, dz_f=increments_back_forward(pID,junk)
        err_table=np.zeros([nfibs,np.max(morphComp[pID+dz_f])]); #table of errors i,j. Overwritten at each new pID

        #CREATING ERROR TABLES
        for fID in range(nfibs):
            #Isolating the relevant 'patch in morphological components
            if fib_rec[fID,pID]!=-1: # catching nonexistent fibrils, true in pID>0
                cofI=props[pID,fib_rec[fID,pID],0:2]#centroid of fibril in plane
                index=np.ndarray.flatten(md.search_window(cofI, npix/10, npix)).astype('int')
                compare_me=np.delete(np.unique(np.ndarray.flatten(morphComp[pID+dz_f,index[0]:index[1], index[2]:index[3]]-1) ),0) #find a more neat way to do this. List of indices in next slice to look at.
                for j in compare_me: #going through relevant segments in next slice
                    err_table[fID,j]=err(pID, fib_rec[fID,pID], fib_rec[fID,pID-dz_b], j,dz_b, dz_f, a, b, c)

        #sorted lists of the errors and the pairs of i,j which yield these Errors
        sort_errs=sort_errs=np.sort(err_table, axis=None)
        sort_err_pairs =np.transpose(np.unravel_index(np.argsort(err_table, axis=None), err_table.shape))

        #delete pairs with 0 errors (ie those who are outside the box) and those above the threshold
        delete=np.concatenate((np.where(sort_errs==0)[0],np.where(sort_errs>errorthresh(a,b,c, skip))[0]), axis=0)
        sort_err_pairs=np.delete(sort_err_pairs,delete, axis=0)

        i=0  #Matching up
        while sort_err_pairs.shape[0]>0:
            match=sort_err_pairs[0]  # picks out smallest error match
            fib_rec[match[0], pID+dz_f]=match[1]  # fills in the corresponding fibril recor with this match
            #delete all other occurences of i,j
            deleteme=np.unique(np.ndarray.flatten(np.concatenate((np.argwhere(sort_err_pairs[:,0]==match[0]),np.argwhere(sort_err_pairs[:,1]==match[1]))))).tolist()
            sort_err_pairs=np.delete(sort_err_pairs, deleteme,axis=0)
            i=i+1
        lis.append(nfibs/np.count_nonzero(fib_rec[:,pID+dz_f]<0))
    return lis

#%%---------------------------------------------------------------------------
#.................................4. MAIN FLOW ...........................
#-------------------------------------------------------------------------------

#CREATING HEATMAP OF ABC VALUES
dirResults=r'C:\Users\t97721hr\Dropbox (The University of Manchester)\Fibril Tracking Algorithm\abc_jan_bc_1.5_2.5'

#chooseABC
N=11
a=1;b=np.linspace(1.5,2.5,N);c=b.copy()
np.savetxt(dirResults+"\\values_abc.txt", np.vstack((np.ones(N),b,c)).T) #saves ABC values

# filling the heatmap, value by value
fun_map = np.empty((b.size, c.size))
def make_abc_map(fun_map):
    for i in range(b.size):
        print(i)
        for j in range(c.size):
            random_planes=np.random.choice(np.setdiff1d(np.arange(nplanes-1),junk),5)
            fun_map[i,j] = np.mean(ndropped(a, b[i],c[j], random_planes))
    np.save(dirResults+"\\heatmap_abc", fun_map)

make_abc_map(fun_map)

#%%
#PLOTTING THE HEATMAP OF ABC VALUES

fun_map=np.load(dirResults+"\\heatmap_abc.npy")

fig = plt.figure()
s = fig.add_subplot(1, 1, 1, xlabel='b', ylabel='c')
extent = [ b[0], b[-1],  c[0],  c[-1]];
im = plt.imshow(fun_map.T,extent=extent, origin='lower') #the transpose is because of the row column effect
fig.colorbar(im);
plt.title('1 in how many fibrils dropped. a=1')
plt.savefig(dirResults+"\\abc.png")
#print ((time_s()-start_time)/60)

#%%
#SORTING THE VALUES OF B AND C
sort_pairs=np.vstack(np.unravel_index((-fun_map).argsort(axis=None, kind='mergesort'), fun_map.shape))
b_c_values_sorted=np.vstack((b[sort_pairs[0]],c[sort_pairs[1]])).T
np.savetxt(dirResults+ "\\a1_b_c_values_sorted.txt", b_c_values_sorted)

##%%
#RUNNING THE BEST VALUES OF ABC AND EXPORTING

for i in range(5):
    a=1
    b,c=b_c_values_sorted[i]
    abc_string="_rank_%d_a_%.2f_b_%.2f_c_%.2f"%(i, a, b, c)
    fibril_mapping(a, b, c, fib_rec_filename='\\fibrec'+abc_string)
    #print('completed %d of 10 rounds of b,c '%i+1)
