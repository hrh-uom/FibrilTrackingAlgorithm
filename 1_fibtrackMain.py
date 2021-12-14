import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation, rc, colors
from PIL import Image
import glob
from skimage.measure import label, regionprops,regionprops_table
from skimage.color import label2rgb
from time import time as time_s
from importlib import reload # reload(module_of_choice);
import customFunctions as md
from feret_diameter import feret_diameters_2d
import os

#-----------------------1. USER INPUT ---------------------------------------------------

start_plane=0; skip=1
end_plane=695 if ('t97721hr' in os.getcwd()) else 5
dirResults, dir3V=md.getDirectories(start_plane, end_plane)
print (dir3V+'\n'+dirResults)

#--------------------------2. IMPORT IMAGES , LABEL, MEASURE-------------------------------------
def create_binary_stack(dir3V, start_plane,end_plane, whitefibrils=True):
    """
    imports images from given 3V directory
    has the option to switch based on whether the fibrils are black on a white background or vice versa
    """
    imagePath = sorted(glob.glob(dir3V+'*.tiff'))[start_plane:end_plane]
    imgstack = np.array( [np.array(Image.open(img).convert('L'), dtype=np.uint16) for img in imagePath])/255
    if whitefibrils==True:
        return imgstack
    else:
        return np.logical_not(imgstack).astype(int)  #May not always be necessary to invert!
    print("Stack Created")
def compress_by_skipping(skip):
    global imgstack
    if skip>1: #Resize array and renumber junk slices if skipping slices
        keep=skip*np.arange(nplanes/skip).astype(int)
        imgstack=imgstack[keep]
        junk=junk/skip
        dz*=skip
        nplanes=imgstack.shape[0]
        dirResults= dir3V+'skip_%d_results/'%skip
def create_morph_comp(imgstack):
    """
    a 3d Labelled array of image stack. Named for the morphological components function in mathematica.
    In each plane, every object gets a unique label, labelled from 1 upwards. The background is labelled 0.
    """
    MC=np.zeros([nplanes, npix, npix], dtype=np.int16) #initialising array for morph comp
    for i in range(nplanes):
        print(f'MC plane {i}')
        MC[i]=label(imgstack[i])
    np.save(dirResults+'morphComp', MC)
    return MC
def create_properties_table(MC):
    """
    Setting up table of properties for each plane (props) props stores (pID, objectID, property).
    it is the length of the max number of objects in any plane, and populated with zeroes.
    Everything is measured in pixels
    """
    props_ofI='centroid','orientation','area','eccentricity' # these properties are the ones to be calculated using skimage.measure
    props=np.empty((nplanes, np.max(MC), len(props_ofI)+2)) #the +2 is because centroid splits into 2, and also to leave space for the feret diameter, calculated by a custom script.
    for pID in range(nplanes):
        print(f'Properties table plane {pID}')
        rprops=pd.DataFrame(regionprops_table(MC[pID], properties=props_ofI)).to_numpy() #regionprops from skimage.measure
        #print (temp.shape)
        nobj=rprops.shape[0]; # nobjects in plane
        props[pID,0:nobj, 0:5]=rprops
        props[pID,0:nobj, 5]=feret_diameters_2d(MC[pID])
    np.save(dirResults+'props', props)
    return props
    #return temp.shape

#Read Metadata File
meta_frame=pd.read_csv(glob.glob(dir3V+'/*metadata*csv')[0])
pxsize=meta_frame.pixelsize[0];junk=meta_frame.junkslices; dz=meta_frame.dz[0]

#Check for previous MC/Properties tables
if (os.path.isfile(dirResults+'morphComp.npy') & os.path.isfile(dirResults+'props.npy')):
    fromscratch=False
else:
    md.create_Directory(dirResults)
    fromscratch=True
    if (os.path.isfile(dirResults+'morphComp.npy') & os.path.isfile(dirResults+'props.npy')): #to save time
        print(f'Loading MC/Props from {dirResults}')
        MC=np.load(dirResults+"morphComp.npy")
        props=np.load(dirResults+"props.npy")
        nplanes, npix, _=MC.shape
    else:
        print("Creating MC/Props from scratch")
        imgstack=create_binary_stack(dir3V, start_plane,end_plane) #import images and create binary array
        nplanes, npix, _ = imgstack.shape #measure array dims
        if skip>1:
            compress_by_skipping(skip)
        MC=create_morph_comp(imgstack)
        props=create_properties_table(MC)

    return pxsize, junk, dz, nplanes, npix, MC, props


#%%-----------------------------------3. FIBRIL MAPPING-------------------------------------------

#----Errors
def err_c(pID, i,prev_i, j, dz_b, dz_f, Lscale):
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
def err_f(pID, i, j, dz_f,Lscale): #error in feret diameter
 return np.abs(props[pID+dz_f, j, 5]-props[pID, i, 5])/Lscale
def err(pID, fID, prev_i, j,dz_b, dz_f, a, b, c):  #not ensuring values need to ve <1
    Lscale=np.median(np.ravel(props[:,:,5])) # A typical lengthscale, calculated from the median value of feret diameter, in pixels. To find the equiv in nm, multiply by pxsize
    return (1/(a+b+c)) *(a* err_c(pID, fID, prev_i,j,dz_b, dz_f, Lscale)+b*err_f(pID, fID, j,dz_f, Lscale)+c*err_a(pID, fID, j,dz_f))
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
def lastplane_tomap(junk,nplanes):
    """
    finding the penultimate plane, incase the last / penultimate is a junk slice
    """
    pID=nplanes
    while np.any(junk==pID):
     pID-=1
    return pID-increments_back_forward(pID, junk)[0]

def initialise_fibril_record(MC):
    nplanes=MC.shape[0]
    nfibs=len(np.unique(MC[0]))-1 #number eqivalent to n objects in first slice
    FR_local=np.full((nfibs,nplanes),-1, dtype=np.int16)  #-1 means no fibril here, as indices are >= 0
    FR_local[:,0]=np.unique(MC[0])[1:]-1  #use like FR_local[fID, pID]
    #return FR_local
    return FR_local
def fibril_mapping(a,b,c, FR_local, skip=1, frfilename='fibrec'):
    """
    Populates fibril record, from top plane through the volume
    """
    start_time=time_s()
    nfibs, nplanes=FR_local.shape
    with open(dirResults+'fibtrack_status_update.csv', 'a') as status_update:
        status_update.write('\ntime '+md.t_d_stamp()+'\nJunk slices,'+str(junk)+"\npID,nfibs,time,time since mapping began (min)")

    for pID in range (lastplane_tomap(junk, nplanes)):
        print(f'Mapping, pid {pID}, t={(time_s()-start_time)/60)}m')
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

                index=np.ndarray.flatten(md.search_window(cofI, 500/pxsize, npix)).astype('int')
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
        with open(dirResults+r'fibtrack_status_update.csv', 'a') as status_update:
            status_update.write('\n'+','.join(map(str,[pID,nfibs,md.t_stamp(),(time_s()-start_time)/60])))
        np.save(dirResults+frfilename, FR_local)
    print(f"mapping complete in {(time_s()-start_time)/60} mins")
    return FR_local

#%%--------------------------4. MAIN FLOW --------------------------------------------------
pxsize, junk, dz, nplanes, npix, MC, props=setup_run()
def main(a, b, c):
    print(f'Mapping plane {start_plane} to {end_plane}')
    FR_core=initialise_fibril_record(MC)

    FR_core=fibril_mapping(a, b, c,FR_core)
main(1,1,1)

#%%---------------------------------------------------------------------------
#.................................5. ABC ROUTINE ...........................
#-------------------------------------------------------------------------------

def ndropped(a, b, c, pID_list):
    """
    figuring out how many dropped in a pair of planes
    """
    lis=[];
    for pID in pID_list:
        nfibs=np.max(MC[pID]);
        fib_rec=np.full((nfibs,nplanes),-1, dtype=int) #-1 means no fibril here, as indices are >= 0
        fib_rec[:,pID]=np.arange(nfibs); #use like fib_rec[fID, pID]
        dz_b, dz_f=increments_back_forward(pID,junk)
        err_table=np.zeros([nfibs,np.max(MC[pID+dz_f])]); #table of errors i,j. Overwritten at each new pID

        #CREATING ERROR TABLES
        for fID in range(nfibs):
            #Isolating the relevant 'patch in morphological components
            if fib_rec[fID,pID]!=-1: # catching nonexistent fibrils, true in pID>0
                cofI=props[pID,fib_rec[fID,pID],0:2]#centroid of fibril in plane
                index=np.ndarray.flatten(md.search_window(cofI, npix/10, npix)).astype('int')
                compare_me=np.delete(np.unique(np.ndarray.flatten(MC[pID+dz_f,index[0]:index[1], index[2]:index[3]]-1) ),0) #find a more neat way to do this. List of indices in next slice to look at.
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
def make_abc_map(a, b, c, N, nrepeats=5):
    # filling the heatmap, value by value
    np.savetxt(dirResults+"/abc/values_abc.txt", np.vstack((np.ones(N),b,c)).T) #saves ABC values
    fun_map = np.empty((b.size, c.size))
    for i in range(b.size):
        for j in range(c.size):
            print(f'i,j={i},{j}')
            random_planes=np.random.choice(np.setdiff1d(np.arange(nplanes-1),junk),nrepeats)
            fun_map[i,j] = np.mean(ndropped(a, b[i],c[j], random_planes))
    np.save(dirResults+"/abc/heatmap_abc", fun_map)
def plot_abc_map(a, b, c):
    #PLOTTING THE HEATMAP OF ABC VALUES
    fun_map=np.load(dirResults+"/abc/heatmap_abc.npy")
    fig = plt.figure()
    s = fig.add_subplot(1, 1, 1, xlabel='b', ylabel='c')
    extent = [ b[0], b[-1],  c[0],  c[-1]];
    im = plt.imshow(fun_map.T,extent=extent, origin='lower') #the transpose is because of the row column effect
    fig.colorbar(im);
    plt.title('1 in how many fibrils dropped. a=1')
    plt.savefig(dirResults+"/abc/abc.png")
    plt.show()
    #print ((time_s()-start_time)/60)
def sort_abc(a, b, c):
    fun_map=np.load(dirResults+"/abc/heatmap_abc.npy")
    #SORTING THE VALUES OF B AND C
    sort_pairs=np.vstack(np.unravel_index((-fun_map).argsort(axis=None, kind='mergesort'), fun_map.shape))
    bcSort=np.vstack((b[sort_pairs[0]],c[sort_pairs[1]])).T
    np.savetxt(dirResults+ "/abc/a1_b_c_values_sorted.txt", bcSort)
    return bcSort
def map_best_abc(bcSort, rank=1):
    #RUNNING THE BEST VALUES OF ABC AND EXPORTING
    print(f'Mapping {rank+1} rank 1 a=1, b={bcSort[rank,0]:.2f}, c={bcSort[rank,1]:.2f}')
    a=1
    b,c=bcSort[rank]
    abc_string="_rank_%d_a_%.2f_b_%.2f_c_%.2f"%(rank, a, b, c)
    fibril_mapping(a, b, c, MC, initialise_fibril_record(MC),frfilename='/abc/fibrec'+abc_string)
def main_abc():
    #CREATING HEATMAP OF ABC VALUES
    N=21
    a=1;b=np.linspace(0,3,N);c=b.copy()
    if os.path.isfile(dirResults+'/abc/a1_b_c_values_sorted.txt'):
    # if os.path.isfile('/Users/user/Dropbox (The University of Manchester)/fibril-tracking/nuts-and-bolts/csf-output/abc-dec21/a1_b_c_values_sorted.txt'):
        print ("Importing abc values from previous run")
        bcSort=np.loadtxt(dirResults+ 'abc/a1_b_c_values_sorted.txt')
        # bcSort=np.loadtxt('/Users/user/Dropbox (The University of Manchester)/fibril-tracking/nuts-and-bolts/csf-output/abc-dec21/a1_b_c_values_sorted.txt')
    else:
        print ("ABC Routine")
        print ("Making ABC map")
        make_abc_map(a, b, c, N, nrepeats=10)
        print ("Plotting ABC map")
        plot_abc_map(a, b, c)
        bcSort=sort_abc(a, b, c)
    print ("Running best ABC values")
    for i in range (5):
        map_best_abc(bcSort, rank=i)

# main_abc()

#
# bcSort=np.loadtxt(dirResults+ 'abc/a1_b_c_values_sorted.txt')
# bcSort.shape
# bcSort[np.all(bcSort<2, axis=1)]

#%%-----------------------------5. ALGORITHM FUNCTION FIG ------------------------

def make_schematic():
    ministack=create_binary_stack(dir3V, 0,2, whitefibrils=True)
    labels=10*ministack[0]
    pID=0; fID=892
    MC[pID].shape
    labels=np.where(MC[pID]==fID+1,fID+1, labels)
    labels.shape
    np.unique(labels)
    cols = ['grey', 'red', 'orange', 'blue', 'yellow', 'purple']
    rgblabel=label2rgb(labels-1, bg_label=-1, colors=cols);
    cofI=props[pID, fID, 0:2]
    xy=md.search_window(cofI, npix/10, npix)[:,0].tolist();
    recsize=np.ndarray.flatten(np.diff(md.search_window(cofI, npix/10, npix))).tolist();
    fig1, ax1 = plt.subplots( )
    ax1.imshow(rgblabel, origin='lower', interpolation='nearest')
    # plt.title('fID %i. Plane %i of %i. Size %i' % (fID,pID+1, nplanes, npix/10))
    # plt.ylabel('y pix')
    # plt.xlabel('x pix')
    # Create a Rectangle patch
    import matplotlib.patches as patches

    rect=patches.Rectangle((xy[1],xy[0]),recsize[0],recsize[1],linewidth=1,edgecolor='w',facecolor='none')
    rect2=patches.Rectangle((xy[1],xy[0]),recsize[0],recsize[1],linewidth=1,edgecolor='w',facecolor='none')
    # Add the patch to the Axes
    ax1.add_patch(rect)
    ax1.set_title("Plane $p$")
    plt.savefig(dirResults+'window-schematic1.png');    plt.show()

    fig2, ax2 = plt.subplots( )

    ax2.set_title("Plane $p+1$")

    index=np.ndarray.flatten(md.search_window(cofI, npix/10, npix)).astype('int')
    compare_me=np.delete(np.unique(np.ndarray.flatten(MC[pID+1,index[0]:index[1], index[2]:index[3]]-1) ),0)
    labels2=10*ministack[1]
    for fID in compare_me:
        labels2=np.where(MC[pID+1]==fID+1,50, labels2)
    rgblabel2=label2rgb(labels2-1, bg_label=-1, colors=cols);
    ax2.imshow(rgblabel2, origin='lower', interpolation='nearest')
    ax2.add_patch(rect2)
    plt.savefig(dirResults+'window-schematic2.png');    plt.show()
# make_schematic()

#%%---------------------------6. ERROR THRESHOLD FIG -------------------------------
def make_errorthresh_fig():
    pID=0;prev_i=100; a,b,c=1,1,1 ;dz_b, dz_f=increments_back_forward(pID, junk)
    ni=np.max(MC[pID]);nj=np.max(MC[pID+1])

    fig, ax = plt.subplots(figsize=[8, 6])
    err_grid_all=np.zeros((ni, nj))
    axins = ax.inset_axes([0.1, 0.56, 0.4, 0.4])

    for i in np.sort(np.random.randint(0,ni,10)):
        cofI=props[pID, i, 0:2]
        index=np.ndarray.flatten(md.search_window(cofI, npix/10, npix)).astype('int')
        compare_me=np.delete(np.unique(np.ndarray.flatten(MC[pID+dz_f,index[0]:index[1], index[2]:index[3]]-1) ),0) #find a more neat way to do this. List of indices in next slice to look at.
        err_grid_window=np.zeros((1, compare_me.size))

        for j in range (nj):
            error=err(pID, i, 0, j,dz_b, dz_f, a, b, c)
            err_grid_all[i,j]=error

        for j in range(len(compare_me)):
            error=err(pID, i, 0, compare_me[j],dz_b, dz_f, a, b, c)
            # print(error)
            err_grid_window[0,j]=error
        # print(cofI, index, compare_me)

        sort_errs=np.sort(err_grid_all, axis=None)
        sort_errs_window=np.sort(err_grid_window, axis=None)
        sort_err_pairs =np.transpose(np.unravel_index(np.argsort(err_grid_all, axis=None), err_grid_all.shape))
        #delete pairs with 0 errors (ie those who are outside the box) and those above the threshold
        delete=np.concatenate((np.where(sort_errs==0)[0],np.where(sort_errs>errorthresh(a,b,c, skip))[0]), axis=0)
        sort_err_pairs=np.delete(sort_err_pairs,delete, axis=0)
        sort_errs_window=sort_errs_window[sort_errs_window>0]
        ax.plot(np.arange(50), sort_errs_window[0:50], label=i)
        axins.plot(np.arange(50), sort_errs_window[0:50], label=i)

    #Set inset region
    x1, x2, y1, y2 = 0,5.1,0,1.3
    axins.set_xlim(x1, x2); axins.set_ylim(y1, y2)

    axins.set_xticks([0,1,2,3,4,5]);axins.set_yticks([0,1])
    ax.set_xlim(0, 50); ax.set_ylim(0, np.around(ax.yaxis.get_data_interval()[1]*1.2,1))
    axins.plot([0, 120], [1, 1], '--k')
    ax.plot([0, 120], [1, 1], '--k')
    ax.set_xlabel("Rank of matched pair"); ax.set_ylabel("Error \u03BE")
    ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.savefig(dirResults+'/stats/error_thresh_fig')
    plt.show()
# make_errorthresh_fig()
