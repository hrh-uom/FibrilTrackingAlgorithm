from a0_initialise import *
import customFunctions as md
from feret_diameter import feret_diameters_2d
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import animation, rc, colors
from skimage.measure import label, regionprops,regionprops_table
from skimage.color import label2rgb
from importlib import reload # reload(module_of_choice);
import os, glob
from datetime import datetime, timedelta
from time import time as tt

#%%-----------------------1. FIBRIL MAPPING-------------------------------------
#------Errors

def err_c(pID, i,prev_i, j, dz_b, dz_f):
    """
    centroid error
    """
    Lscale=props[pID, i, 5] #A sensible lengthscale for this fibril to move by. IE, its diameter.
    #TEMPORARY !!! VERTICAL MAPPING
    if pID!=-1: #top slice, no history.
        return np.linalg.norm(props[pID, i, 0:2]-props[pID+dz_f, j, 0:2])/Lscale
    else:
        currentcent=props[pID, i, 0:2]
        prevcent=props[pID-dz_b, prev_i, 0:2]
        predictedcent=currentcent+dz_f*(currentcent-prevcent)
        return np.linalg.norm(predictedcent-props[pID+dz_f, j, 0:2])/Lscale
def err_a(pID, i, j, dz_f): #error in area
    Ascale=props[pID, i, 3]
    return np.abs(props[pID+dz_f, j, 3]-props[pID, i, 3])/Ascale
def err_f(pID, i, j, dz_f): #error in feret diameter
    Lscale=props[pID, i, 5] #A sensible lengthscale for this fibril to move by. IE, its diameter.
    return np.abs(props[pID+dz_f, j, 5]-props[pID, i, 5])/Lscale
def err(pID, fID, prev_i, j,dz_b, dz_f, a, b, c):  #not ensuring values need to ve <1
    return (1/(a+b+c)) *(a* err_c(pID, fID, prev_i,j,dz_b, dz_f)+b*err_f(pID, fID, j,dz_f)+c*err_a(pID, fID, j,dz_f))
def errorthresh(a, b, c, skip): #max values for error cutoff.
 return (1/(a+b+c))*(a *skip + b + c )
#--------Junk Slice Functions
def increments_back_forward(pID):
    """
    skipping slices in the case of junk slices
    """
    test_plane=pID
    inc_fw=1
    while np.any(d.junk==test_plane+1):
     inc_fw=inc_fw+1
     test_plane=test_plane+1
    test_plane=pID
    inc_back=1
    while np.any(d.junk==test_plane-1):
     inc_back=inc_back+1
     test_plane=test_plane-1
    return inc_back,inc_fw
def lastplane_tomap():
    """
    finding the penultimate plane, incase the last / penultimate is a junk slice
    """
    pID=d.nP
    while np.any(d.junk==pID):
     pID-=1
    return pID-increments_back_forward(pID)[0]

#--------Populating
def initialise_fibril_record(MC):
    nfibs=len(np.unique(MC[0]))-1 #number eqivalent to n objects in first slice
    FR_local=np.full((nfibs,d.nP),-1, dtype=np.int16)  #-1 means no fibril here, as indices are >= 0
    FR_local[:,0]=np.unique(MC[0])[1:]-1  #use like FR_local[fID, pID]
    #return FR_local
    return FR_local
def fibril_mapping(a,b,c, MC, FR_local, skip=1, FRFname='fib_rec'):

    """
    Populates fibril record, from top plane through the volume
    """
    start_time=tt()
    nfibs=FR_local.shape[0]
    md.print_status('\npID,nfibs,time,time since mapping began (min)')
    for pID in range (lastplane_tomap()):
        print(f'Mapping, pid {pID}, t={str(timedelta(seconds=np.round(tt()-start_time)))}')
        if np.any(d.junk==pID):#If the slice is junk, skip the mapping.
            #x=1
            continue
        dz_b, dz_f=increments_back_forward(pID)
        err_table=np.zeros([nfibs,np.max(MC[pID+dz_f])])  #table of errors i,j.Overwritten at each new pID
        # print("waypoint1")
        #CREATING ERROR TABLES
        for fID in range(nfibs):
            # print(f"fib ID {fID}")
            #Isolating the relevant 'patch in morphological components
            if FR_local[fID,pID]!=-1: # catching nonexistent fibrils, true in pID>0
                cofI=props[pID,FR_local[fID,pID],0:2]#centroid of fibril in plane
                index=np.ndarray.flatten(md.search_window(cofI, d.npix/10, d.npix)).astype('int')

                compare_me=np.delete(np.unique(np.ndarray.flatten(MC[pID+dz_f,index[0]:index[1], index[2]:index[3]]-1) ),0) #find a more neat way to do this. List of indices in next slice to look at.
                for j in compare_me: #going through relevant segments in next slice
                    # print(f"Compare me {j}")
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
        fibrec_append=np.full((new_objects.size,d.nP),-1, dtype=int)  #an extra bit to tack on the end of the fibril record accounting for all these new objects
        fibrec_append[:,pID+dz_f]=new_objects
        FR_local=np.concatenate((FR_local,fibrec_append))
        nfibs+=new_objects.size

        # save/export stuff
        md.print_status('\n'+','.join(map(str,[pID,nfibs,datetime.now(), timedelta(seconds=np.round(tt()-start_time))], )))
        np.save(d.dirOutputs+FRFname, FR_local)
    print('mapping complete in'+ str(timedelta(seconds=np.round(tt()-start_time))))
    return FR_local

#%%------------------------------2. MAIN FLOW -----------------------------------
if __name__ == "__main__":

    a,b,c=1,1,1

    # d=dataset(input('Enter Dataset'))
    md.print_status('\n\nNEW RUN \n time,'+str(datetime.now())+f'\n a, {a}\n b,{b} \n c,{c}\n')
    for key, value in vars(d).items():
        print(key,'\t',value)
        md.print_status(key+','+str(value)+'\n')

    MC, props=setup_MC_props()
    FR_core=initialise_fibril_record(MC)
    FR_core=fibril_mapping(a, b, c, MC,FR_core)

# main(1,1,1)

#%%--------------------------SANDBOX--------------------------------------------
