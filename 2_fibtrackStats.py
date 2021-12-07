import numpy as np
import matplotlib.pyplot as plt
import customFunctions as md
from scipy import stats
import glob
import os
import pandas as pd
plt.style.use('./mystyle.mplstyle')

#----------------------------------------------------------------------------
#.....................................USER INPUT.............................
#-------------------------------------------------------------------------------
desired_length=1000 #nm
start_plane,end_plane=0,695
abc=False

dirResults, dir3V=md.getDirectories(start_plane, end_plane)
md.create_Directory(dirResults+'stats/')
#----------------------------------------------------------------------------
#....................IMPORT DATA FROM FIBRIL MAPPING....................
#------------------------------------------------------------------------------

try:
    fib_rec_0=np.load(dirResults+'fib_rec.npy') #original, import fibril record
    morphComp=np.load(dirResults+'morphComp.npy')
    props=np.load(dirResults+'props.npy')
except:
    print("Error, no fibrec, morphComp, props found")

if abc:
    fib_rec_0=np.load('/Users/user/Dropbox (The University of Manchester)/fibril-tracking/nuts-and-bolts/csf-output/abc-dec21/rank0/fibrec_rank_0_a_1.00_b_2.70_c_2.85.npy')
    dirResults='/Users/user/Dropbox (The University of Manchester)/fibril-tracking/nuts-and-bolts/csf-output/abc-dec21/rank0/'

print("Testing")
#-~~~~~~~~~~~~MEASURE SHAPE AND SIZE~~~~~~~~~~~~
nplanes, npix, npix=morphComp.shape
#Read Metadata File
meta_frame=pd.read_csv(glob.glob(dir3V+'*metadata*csv')[0])
pxsize=meta_frame.pixelsize[0];junk=meta_frame.junkslices; dz=meta_frame.dz[0]
frac=np.round(desired_length/pxsize)/nplanes
md.create_Directory(dirResults+'stats')
#-~~~~~~~~~~~~TRIM IF NEEDED ~~~~~~~~~~~~
if os.path.isfile(dirResults+f'fib_rec_trim_{frac:.2f}.npy'):
    print("Loading")
    fib_rec=np.load(dirResults+f'fib_rec_trim_{frac:.2f}.npy') #original, import fibril record
else:
    print("Trimming")
    fib_rec=md.trim_fib_rec(fib_rec_0, morphComp, dirResults, frac)

nfibs=fib_rec.shape[0]

#%%----------------------------------------------------------------------------
#....................GEOMETRY OF FIBRIL POPULATION....................
#-------------------------------------------------------------------------------

def fascicleCoord(pID):
    """
    Calculates the mean co-ordinate of all fibrils at slice pID
    """
    objects_in_plane=fib_rec[:, pID][ fib_rec[:, pID]>-1]
    if objects_in_plane.size!=0: #ignoring junk slices
        coOrds_2D=[]
        for i in objects_in_plane:
            coOrds_2D.append((props[pID, i, 0:2]*pxsize))
        return np.append(np.mean(np.array(coOrds_2D), axis=0), dz*pID)
    else:
        return np.array([-1,-1, -1])
def fibCoords(i):
    """
    calculates the coordinates in 3d real space of a fibril (i).
    """
    co_ords=np.full((nplanes, 3),-1.)  #an array of the centroid co-ordinates for each fibril
    for pID in range(nplanes):
        if fib_rec[i, pID]!=-1:
            co_ords[pID, 0:2]=(props[pID, fib_rec[i,pID], 0:2])*pxsize
            co_ords[pID, 2]=pID*dz
    co_ords_trim=co_ords[co_ords[:,2]>-1.]  #getting rid of junk slices / places where absent
    #This stuff is to draw a line of best fit! Leaving it out
    #mean = np.mean(co_ords, axis=0)
    #uu, dd, vv=np.linalg.svd(co_ords-mean)
    #direction=vv[0]
    return co_ords_trim,co_ords #, mean, direction

def coOrds_to_length(co_ords):
    """
    input an Nx3 numpy array representing a list of 3D coordinates, and it will calculate the length of the 'worm like' length joining all the co-ordinates in 3d
    """
    L=0
    for j in range((co_ords.shape[0])-1): #j steps through planes in co-ords
        dr=co_ords[j]-co_ords[j+1]
        L+=np.linalg.norm(dr)
    return L
junk

def calculate_fascicle_length():
    """
    Calculate the arclength of the fascile
    """
    fas_coord_list=[]
    for pID in range (nplanes-1):
        if np.isin(pID, junk)==False:
            fas_coord_list.append(fascicleCoord(pID))
    fas_len=coOrds_to_length(np.array(fas_coord_list))
    return fas_len, fas_coord_list
fas_len, fas_coord_list=calculate_fascicle_length()

#Calculate length of each fibril
#Q: How long are all the fibrils in the fibril rec?
nexist=np.zeros(nfibs, dtype='int')
for i in range(nfibs):
    nexist[i]=np.max(np.nonzero(fib_rec[i]>-1))-np.min(np.nonzero(fib_rec[i]>-1))+1


lengths_scaled=np.zeros(nfibs)  #worm like length / n planes present
for i in range (nfibs):
    lengths_scaled[i]=coOrds_to_length(fibCoords(i)[0])
lengths_scaled*=nplanes/(fas_len*nexist)
#How long are the long fibrils?
print(f'Critical Strain. Fibril strands appear in {desired_length/1000}um of z distance')
md.my_histogram((lengths_scaled), 'Length relative to fascicle', binwidth=.005,filename=dirResults+f'stats/CS_dist_{desired_length}nm_{frac:.2f}.png')
np.save(dirResults+f'stats/scaledlengths_{desired_length}nm_{frac:.2f}', lengths_scaled)

#%% WHERE IS THE FASCICLE GOING
import matplotlib.cm as cm
from matplotlib.collections import LineCollection


fig, ax=plt.subplots(figsize=(10,10))

fas_arr=np.array(fas_coord_list)
x=fas_arr[:,0]/pxsize
y=fas_arr[:,1]/pxsize
cols = np.linspace(0,100*(1+nplanes//100),len(x))



ax.set_aspect(1)
ax.set_xlim(450,550);ax.set_ylim(420, 520)
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
ax.set_xlabel('x (pixels)');ax.set_ylabel('y (pixels)')
lc = LineCollection(segments, cmap='gist_rainbow')
lc.set_array(cols); lc.set_linewidth(2)
line = ax.add_collection(lc)
fig.colorbar(line,ax=ax)
plt.savefig(dirResults+'stats/fascicle-travel.png',bbox_inches='tight' ,pad_inches = 0);plt.show()




#%%---------------------------Feret Diameter of each fibril

def fibril_MFD(i, FR): #maps between props and fibrec
    feret_planewise=np.full(nplanes,-1.)  #an array of the centroid co-ordinates for each fibril
    for pID in range(nplanes):
        if FR[i, pID]!=-1:
            feret_planewise[pID]=(props[pID, FR[i,pID], 5])*pxsize
    feret_planewise=feret_planewise[feret_planewise>-1.]  #getting rid of junk slices / places where absent
    mean = np.mean(feret_planewise, axis=0)
    return mean,feret_planewise

fib_MFDs=np.array([fibril_MFD(i, fib_rec)[0] for i in range(nfibs)])
np.save(dirResults+f'fib_MFDs_{desired_length}nm_{frac:.2f}', fib_MFDs)
md.my_histogram(fib_MFDs, 'Minimum Feret Diameter (nm)', 'Minimum Feret Diameter distribution', filename=dirResults+f'stats/MFD_dist_{desired_length}nm_{frac:.2f}.png')

#%% ------------------------Area of each fibrils

def fibril_area(i):
    """
    Delivers fibril area in nm^2, for some fibril in the Fibril Record i
    """
    area_planewise=np.full(nplanes,-1.)  #an array of the areas for each fibril
    for pID in range(nplanes):
        if fib_rec[i, pID]!=-1:
            area_planewise[pID]=(props[pID, fib_rec[i,pID],3])*(pxsize**2)
    area_planewise=area_planewise[area_planewise>-1.]  #getting rid of junk slices / places where absent
    mean = np.mean(area_planewise)
    return mean, area_planewise
fibrilArea=np.array([fibril_area(i)[0] for i in range(nfibs)])
np.save(dirResults+f'area_{desired_length}nm_{frac:.2f}.npy', fibrilArea)
md.my_histogram(fibrilArea/100, 'Area ($10^3$ nm$^2$)', 'Cross Sectional Area of tracked fibrils', binwidth=50)
#%%----------------Orientation


# from importlib import reload ; reload(md)
# def fib_ori(i, FR): #maps between props and fibrec
#     ecc_planewise=np.full(nplanes,-1.)  #an array of the centroid co-ordinates for each fibril
#     for pID in range(nplanes):
#         if FR[i, pID]!=-1:
#             ecc_planewise[pID]=(props[pID, FR[i,pID], 2])
#     ecc_planewise=ecc_planewise[ecc_planewise>-1.]  #getting rid of junk slices / places where absent
#     mean = np.mean(ecc_planewise, axis=0)
#     return mean,ecc_planewise
# fib_oris=(180/np.pi)*np.array([fib_ori(i, fib_rec)[0] for i in range(nfibs)])

fas_coords=np.array([fascicleCoord(pID) for pID in range (nplanes)]) #This includes junk planes
anglelis=[]
for fID in range(nfibs):
    fib_coords_values=fibCoords(fID)[1]
    for pID in np.arange(nplanes-1):
        if np.all(np.stack((fib_coords_values[pID], fib_coords_values[pID+1]))>=0):
            a=fas_coords[pID+1]-fas_coords[pID]
            b=fib_coords_values[pID+1]-fib_coords_values[pID]
            angle=(180/(2*np.pi))*np.arccos(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))
            if angle>180:
                angle=angle-180
            anglelis.append(angle)
            # print(f'fid {fID} pID {pID}')

md.my_histogram(np.array(anglelis), 'Direction relative to mean (degrees)',  dens=True,binwidth=2, pi=False,filename=dirResults+'/stats/orientation.png')


#%%----------------Length vs cross secitonal Area
fib_MFDs
plt.plot(lengths_scaled,fibrilArea/10**6, '.r')
plt.xlabel('Normalised lengths')
plt.ylabel('Cross Sectional Area (um$^2$)')
plt.show()

#%%WHAT ARE THE REALLY LONG FIBRILS UP to
fig, (ax1, ax2)=plt.subplots(1,2)
XLfibs=np.where(lengths_scaled>1.1)[0]
ax1.set_xlabel('Normalised lengths')
ax1.set_ylabel('Minimum Feret Diameter (nm)')
ax1.plot(lengths_scaled[XLfibs],fib_MFDs[XLfibs], '.r', )
ax2.set_xlabel('Minimum Feret Diameter (nm)')
ax2.set_ylabel('Density')
ax2.hist(fib_MFDs[XLfibs], density=True)
fig.tight_layout()
plt.savefig(dirResults+'stats/xl-fibs.png')
plt.show()
xl_fibs_render=False
if xl_fibs_render:
    labels=md.label_volume(morphComp, XLfibs, fib_rec, 695)
    md.export_animation(dirResults, XLfibs, labels, title='stats/xl-fibrils-animation', step=5)
    md.volume_render(labels, 0, nplanes, 0, 1000,pxsize, dz,dirResults,'stats/XLvolumerender')
#%% d1 fibrils
def d1_3d():
    d1_fibrils=np.where(fib_MFDs<100)[0]
    dirResults
    labels=md.label_volume(morphComp, d1_fibrils, fib_rec, 695)
    md.export_animation(dirResults, d1_fibrils, labels, title='stats/D1-animation', step=5)
    md.volume_render(labels, 0, nplanes, 0, 1000,pxsize, dz,dirResults,'stats/D1-render')
#%%----------------------------------------------------------------------------
#....................TESTING FOR STATISTICAL SIGNIFICANCE ....................
#------------------------------------------------------------------------------
#Q: which segments don't get picked up?
junk
planes_no_cells=np.array([1,25,47,85,95,96,111, 129,155,158,277,291, 296])-1 #read off imageJ
tracked_FD=np.ones([1]);untracked_FD=np.ones([1]);

# for pID in planes_no_cells:
#     tracked=np.setdiff1d(np.unique(fib_rec[:,pID]),np.array([-1]))
#     tracked_FD=np.concatenate((tracked_FD,props[pID, tracked, 5]*pxsize))
#     untracked=np.setdiff1d(np.arange(np.count_nonzero(props[0,:,3])),tracked)
#     untracked_FD=np.concatenate((untracked_FD,props[pID, untracked, 5]*pxsize))
for pID in [nplanes//2]:
    trackedIDs=np.unique(fib_rec[:,pID])[1:]
    tracked_FDs=props[pID,trackedIDs, 5]*pxsize
    allIDs=np.unique(morphComp[pID]-1)[1:]
    allFDs=props[pID,allIDs, 5]*pxsize


trackedIDs.shape
allFDs.shape

lower, upper=90,350
rel_all_FD=allFDs[(allFDs>lower) & (allFDs<upper)]
rel_tracked_FD=tracked_FDs[(tracked_FDs>lower) & (tracked_FDs<upper)]

kstest=stats.ks_2samp(rel_tracked_FD, rel_all_FD)
result="reject" if kstest[1]<0.05 else "accept"
title=f'$H_0$, these two samples come from the same distribution. p={kstest[1]:.2f}: {result}\n Distribution limited to ({lower}, {upper}) nm'
print(title)

# md.my_histogram([rel_tracked_FD, rel_all_FD],'Feret diameter (nm)', binwidth=20,cols=['red', 'lime'], dens=False, labels=['Tracked fibrils', 'All segments'],filename=dirResults+f'stats/statistical_significance_CS_dist_{desired_length}nm_{frac:.2f}.png', leg=True)

fig,ax=plt.subplots(figsize=(12, 8))
nbins=13
bins = np.linspace(lower, upper, nbins+1)
ax.hist(rel_tracked_FD, bins, alpha = 1, color='blue',label='Tracked fibrils')

ax.hist(rel_all_FD, bins, alpha = 0.5, color='grey',label='All segments', edgecolor='black')
ax.legend(loc='upper left')
ax.set_ylabel('Number')
ax.set_xlabel('Minimum Feret diameter (nm)')
ax.grid(visible=False,which='major', axis='y')

ax.set_xticks(np.linspace(100, upper,6), minor=False)
ax.set_xticks(np.linspace(100, upper,31), minor=True)

hist_tracked, edges=np.histogram(rel_tracked_FD, bins=nbins, range=(lower, upper))
hist_all, _=np.histogram(rel_all_FD, bins=nbins, range=(lower, upper))
hist_tracked
ax2 = ax.twinx()
nonzero=np.ndarray.flatten(np.argwhere(hist_all!=0))
nonzero
hist_tracked[nonzero].size
hist_all[nonzero].size
ax2.plot((edges+0.5*(edges[1]-edges[0]))[:-1][nonzero],hist_tracked[nonzero]/hist_all[nonzero], '--k')
ax2.set_ylim(0,1.05)
ax2.set_yticks(np.arange(0, 110, 20)/100)
ax2.set_ylabel('Fraction of segments captured')

filename=dirResults+f'stats/statistical_significance_CS_dist_{desired_length}nm_{frac:.2f}.png'
plt.savefig(filename);
plt.show()
#%%----------------------------------------------------------------------------
#....................DROPPED FIBRIL INQUIRIES....................
#-------------------------------------------------------------------------------
#Q: read original fibril record, before chopping.
nfibs_0,nplanes=fib_rec_0.shape

#Q: how many fibs are we capturing in cross-section?
meanperplane=np.mean(np.apply_along_axis(np.max, 1, np.reshape(morphComp, (nplanes,npix**2 ))))
n=0
for i in range(nplanes):
    n+=fib_rec[i][fib_rec[i]>-1].size


print('fraction captured in cross section', (n/nplanes)/meanperplane)

#Q: How long are all the fibrils in the original fibril rec?
nexist_0=np.zeros(nfibs_0, dtype='int')
for i in range(nfibs_0):
    nexist_0[i]=np.max(np.nonzero(fib_rec_0[i]>-1))-np.min(np.nonzero(fib_rec_0[i]>-1))+1


title_=f'Strands: {nfibs_0}, > {desired_length}nm in z: {nfibs} $\sim$ {100*nfibs/nfibs_0:.0f}%'
print(title_)
md.my_histogram(nexist_0/nplanes,'Fraction of planes present', binwidth=.05, filename=dirResults+'stats/planes_present')
md.my_histogram(nexist/nplanes,'Fraction of planes present', binwidth=.05, filename=dirResults+'stats/planes_present_zoom')


##%%---OLD STUFF BELOW ----------------------------------------------------------------------------

#Q: which ones are nearly full length but not quite 6/11/2020
half_length_fibril_indices=np.nonzero((0.5*nplanes<nexist_0)&(nexist_0<=frac*nplanes))[0]
def plot_half_length():
    plt.plot(np.count_nonzero(fib_rec_0[half_length_fibril_indices]>-1, axis=0), '.-b')
    plt.xlabel("Plane (/100)")
    plt.ylabel("Nfibrils")
    plt.grid()
    plt.xticks(np.arange(0,nplanes, 10))
    plt.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    plt.title("Examining the fibrils which appear in 50-90% of all planes")
    plt.show()
# plot_half_length()
##%% Q: where are the fibril ends in the half length group?
def plot_fib_ends_half_length():
    fibril_ends=[]
    for fID in half_length_fibril_indices:
        temp=np.nonzero(fib_rec_0[fID]>-1)[0]
        fibril_ends.extend([temp[0],temp[-1]])
    fibril_ends=np.array(fibril_ends)
    N=fibril_ends.size
    fibril_ends=fibril_ends[fibril_ends>0]
    fibril_ends=fibril_ends[fibril_ends<nplanes]
    counts=0; unique, counts = np.unique(fibril_ends, return_counts=True)
    if np.any (counts): #In prototyping, sometimes the fibril ends group is empty.
        n, bins, patches = plt.hist(fibril_ends, nplanes, density=False, facecolor='g', alpha=0.75)
        plt.ylim(0,max(counts))
        plt.xlabel("Plane");plt.ylabel('Number')
        plt.vlines(junk+0.5, 0, 1000)
        plt.title("Location of Fibril ends, 50% and longer segments excluding 0 and 100")
        plt.grid(True)
        plt.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
        plt.show()
        print("%i out of %i are 0 and 100, thats %lf percent" % (N-fibril_ends.size,N,100* (N-fibril_ends.size)/N))
plot_fib_ends_half_length()

##%% Question: Where are the fibril ends (ALL)
def plot_fib_tops_bottoms():

    f_tops=[];f_bottoms=[]
    for fID in range(nfibs_0):
        temp=np.nonzero(fib_rec_0[fID]>-1)[0]
        f_tops.append(temp[0])
        f_bottoms.append(temp[-1])
    cols=['red', 'lime']
    n, bins, patches = plt.hist(f_tops+f_bottoms, nplanes//10, density=False, histtype='bar')
    #n, bins, patches = plt.hist(f_bottoms, nplanes, density=False, facecolor='b', alpha=0.5)
    plt.xlabel("Plane")
    plt.ylabel('Number')
    print('Where are the fibril ends?')
    unique, counts = np.unique(f_tops+f_bottoms, return_counts=True)
    plt.ylim(0,max(counts)*1.5)
    plt.vlines(junk+0.5, 0, max(counts)*1.5, colors='black')
    plt.xticks(np.arange(0, nplanes, 50))
    plt.grid(True)
    plt.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)

    plt.savefig(dirResults+'stats/fibrilends');plt.show()
plot_fib_tops_bottoms()
#%% is there a correlation between size and nplanes in which the strand exists
def plot_strand_size_vs_MFD():
    fib_MFDs_0=np.array([fibril_MFD(i, fib_rec_0)[0] for i in range(nfibs_0)])
    plt.scatter(fib_MFDs_0,nexist_0, s=8)
    plt.title("Is there a correlation between strand MFD and length of strand")
    plt.ylim(0,102)
    plt.xlim(lower,upper)
    plt.xlabel("Strand MFD (nm)")
    plt.yticks(np.arange(0,nplanes, 10))
    plt.xticks(np.arange(lower,upper, 20))
    plt.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    plt.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    plt.ylabel("Number of planes in which strand exists")
    plt.show()
# plot_strand_size_vs_MFD()
