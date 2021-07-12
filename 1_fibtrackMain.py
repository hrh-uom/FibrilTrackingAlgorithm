import numpy as np
import pandas as pd
from matplotlib import animation, rc, colors
import matplotlib.pyplot as plt
from PIL import Image
import glob
from IPython.display import HTML
from skimage.measure import label, regionprops,regionprops_table
from skimage.color import label2rgb
from time import time as time_s
from importlib import reload # reload(module_of_choice);
import customFunctions as md
from feret_diameter import feret_diameters_2d
plt.style.use('./mystyle.mplstyle')
#----------------------------------------------------------------------------------
#...............................1. USER INPUT .................................
#------------------------------------------------------------------------------------
fromscratch=False
whichdata=0;skip=1
start_plane, end_plane=10, 25
#----------------------------------------------------------------------------------
#.........................2. IMPORT IMAGES , LABEL, MEASURE ......................
#------------------------------------------------------------------------------------
def create_binary_stack(dir3V, start_plane,end_plane):
    """
    imports images from given 3V directory
    """
    imagePath = glob.glob( dir3V + 'fin\\*.tif')[start_plane:end_plane]
    imgstack_neg = np.array( [np.array(Image.open(img).convert('L'), 'uint8') for img in imagePath])
    imgstack=np.logical_not(imgstack_neg).astype(int)  #May not always be necessary to invert!
    return imgstack
def compress_by_skipping(skip):
    global imgstack
    if skip>1: #Resize array and renumber junk slices if skipping slices
        keep=skip*np.arange(nplanes/skip).astype(int)
        imgstack=imgstack[keep]
        junk=junk/skip
        dz*=skip
        nplanes=imgstack.shape[0]
        dirResults= dir3V+'skip_%d_results\\'%skip
def create_morph_comp(imgstack):
    """
    a 3d Labelled array of image stack. Named for the morphological components function in mathematica. In each plane, every object gets a unique label, labelled from 1 upwards. The background is labelled 0.
    """
    MC=np.zeros([nplanes, npix, npix], dtype=int) #initialising array for morph comp
    for i in range(nplanes):
        MC[i]=label(imgstack[i])
    np.save(dirResults+'morphComp', MC)
    return MC
def create_properties_table(MC):
    """
    Setting up table of properties for each plane (props) props stores (pID, objectID, property). it is the length of the max number of objects in any plane, and populated with zeroes.
    """
    props_ofI='centroid','orientation','area','eccentricity' # these properties are the ones to be calculated using skimage.measure
    props=np.empty((nplanes, np.max(MC), len(props_ofI)+2)) #the +2 is because centroid splits into 2, and also to leave space for the feret diameter, calculated by a custom script.
    for pID in range(nplanes):
        rprops=pd.DataFrame(regionprops_table(MC[pID], properties=props_ofI)).to_numpy() #regionprops from skimage.measure
        #print (temp.shape)
        nobj=rprops.shape[0]; # nobjects in plane
        props[pID,0:nobj, 0:5]=rprops
        props[pID,0:nobj, 5]=feret_diameters_2d(MC[pID])
    np.save(dirResults+'props', props)
    return props
    #return temp.shape

dir3V=md.find_3V_data(whichdata); # find the relevant data based on the timepoint desired
pxsize, dz=np.genfromtxt( dir3V+'pxsize.csv', delimiter=',')[1] #import voxel size
junk=pd.read_csv( dir3V+'junkslices.csv', header=None).to_numpy().T[0]-start_plane #which slices are broken
dirResults= dir3V+f'results_{start_plane}_{end_plane}\\' #Create subfolder (if it doesnt already exist!)
md.create_Directory(dirResults)

if fromscratch:
    imgstack=create_binary_stack(dir3V, start_plane,end_plane) #import images and create binary array
    nplanes, npix, _ = imgstack.shape #measure array dims
    if skip>1:
        compress_by_skipping(skip)
    MC=create_morph_comp(imgstack);
    props=create_properties_table(MC)
else: #to save time
    MC=np.load(dirResults+"morphComp.npy")
    props=np.load(dirResults+"props.npy")
    nplanes, npix, _=MC.shape

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

def initialise_fibril_record(MC):
    nfibs=len(np.unique(MC[0]))-1 #number eqivalent to n objects in first slice
    FR_local=np.full((nfibs,nplanes),-1, dtype=int)  #-1 means no fibril here, as indices are >= 0
    FR_local[:,0]=np.unique(MC[0])[1:]-1  #use like FR_local[fID, pID]
    #return FR_local
    return FR_local

def fibril_mapping(a,b,c, MC, FR_local, skip=1, reduction=0, rAnge=lastplane_tomap(junk)):
    """
    Populates fibril record, from top plane through the volume
    """
    start_time=time_s()
    nfibs=FR_local.shape[0]
    with open(dirResults+r'\fibtrack_status_update.csv', 'a') as status_update:
        status_update.write('\ntime '+md.t_d_stamp()+'\nJunk slices,'+str(junk)+"\npID,nfibs,time since mapping began")

    for pID in range (lastplane_tomap(junk)):
        if np.any(junk==pID):#If the slice is junk, skip the mapping.
            #x=1
            continue
        dz_b, dz_f=increments_back_forward(pID,junk)
        err_table=np.zeros([nfibs,np.max(MC[pID+dz_f])])  #table of errors i,j.Overwritten at each new pID

        #CREATING ERROR TABLES
        for fID in range(nfibs):
            #Isolating the relevant 'patch in morphological components
            if FR_local[fID,pID]!=-1: # catching nonexistent fibrils, true in pID>0
                cofI=props[pID,FR_local[fID,pID],0:2]#centroid of fibril in plane
                index=np.ndarray.flatten(md.search_window(cofI, npix/10, npix)).astype('int')
                compare_me=np.delete(np.unique(np.ndarray.flatten(MC[pID+dz_f,index[0]:index[1], index[2]:index[3]]-1) ),0) #find a more neat way to do this. List of indices in next slice to look at.
                for j in compare_me: #going through relevant segments in next slice
                    err_table[fID,j]=err(pID, FR_local[fID,pID], FR_local[fID,pID-dz_b], j,dz_b, dz_f, a, b, c)

        #sorted lists of the errors and the pairs of i,j which yield these Errors
        sort_errs=sort_errs=np.sort(err_table, axis=None)
        sort_err_pairs =np.transpose(np.unravel_index(np.argsort(err_table, axis=None), err_table.shape))

        #delete pairs with 0 errors (ie those who are outside the box) and those above the threshold
        delete=np.concatenate((np.where(sort_errs==0)[0],np.where(sort_errs>errorthresh(a,b,c, skip))[0]), axis=0)
        sort_err_pairs=np.delete(sort_err_pairs,delete, axis=0)

        i=0  #Matching up
        while sort_err_pairs.shape[0]>0:
            match=sort_err_pairs[0]  # picks out smallest error match
            FR_local[match[0], pID+dz_f]=match[1]  # fills in the corresponding fibril recor with this match
            #delete all other occurences of i,j
            deleteme=np.unique(np.ndarray.flatten(np.concatenate((np.argwhere(sort_err_pairs[:,0]==match[0]),np.argwhere(sort_err_pairs[:,1]==match[1]))))).tolist()
            sort_err_pairs=np.delete(sort_err_pairs, deleteme,axis=0)
            i=i+1

        #LOOK FOR MISSED FIBRILS
        all=np.arange(np.max(MC[pID+dz_f]))
        mapped=np.unique(FR_local[:,pID+dz_f][FR_local[:,pID+dz_f]>-1])
        new_objects=np.setdiff1d(all, mapped)
        fibrec_append=np.full((new_objects.size,nplanes),-1, dtype=int)  #an extra bit to tack on the end of the fibril record accounting for all these new objects
        fibrec_append[:,pID+dz_f]=new_objects
        FR_local=np.concatenate((FR_local,fibrec_append))
        nfibs+=new_objects.size

        # save/export stuff
        with open(dirResults+r'\fibtrack_status_update.csv', 'a') as status_update:
            status_update.write('\n'+','.join(map(str,[pID,nfibs,time_s()-start_time])))
        np.save(dirResults+'fib_rec', FR_local)
    return FR_local

def trim_fib_rec(FR_local,frac=0.9):
    """
    Trims fibril record to fibrils which are less than some fraction of the total number of planes
    """
    #Q: How long are all the fibrils in the original fibril rec?
    nfibs=FR_local.shape[0]
    nexist=np.zeros(nfibs, dtype='int')
    for i in range(nfibs):
        nexist[i]=np.max(np.nonzero(FR_local[i]>-1))-np.min(np.nonzero(FR_local[i]>-1))+1
    longfibs=np.where(nexist>nplanes*frac)[0]  #the indices of the long fibirls
    #Erasing fibril record for short fibrils. The only way to map between the two is using longfibs. Reindexing also!
    np.save(dirResults+f'fib_rec_trim_{frac}', FR_local[longfibs])
    np.save(dirResults+f'labelledVol_{frac}',md.label_volume(MC,np.arange(longfibs.size), FR_local[longfibs], nplanes))
    return FR_local[longfibs]

#%%---------------------------------------------------------------------------
#.................................4. MAIN FLOW ...........................
#-------------------------------------------------------------------------------
a,b,c=1,1,1

FR_core=initialise_fibril_record(MC)
FR_core=fibril_mapping(a, b, c, MC,FR_core)
FR_core=trim_fib_rec(FR_core, frac=0.9)
np.save(dirResults+f'labeled_volume', FR_local[longfibs])

md.beep()
#md.animation_inline(MC,np.arange(FR_core.shape[0]), FR_core,0,nplanes)

#%%---------------------------------------------------------------------------
#.................................SANDBOX....................................
#-------------------------------------------------------------------------------
