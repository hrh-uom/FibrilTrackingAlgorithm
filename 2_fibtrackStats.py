import numpy as np
import matplotlib.pyplot as plt
import customFunctions as md
import glob
from scipy import stats
plt.rcParams['figure.figsize'] = [10, 7.5] #default plot size
plt.rcParams['font.size']=16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['savefig.facecolor']='white'

#----------------------------------------------------------------------------
#.....................................USER INPUT.............................
#-------------------------------------------------------------------------------
whichdata=0
desired_length=0.9
resultsDir=md.find_3V_data(whichdata)+'results\\';
#----------------------------------------------------------------------------
#....................IMPORT DATA FROM FIBRIL MAPPING....................
#------------------------------------------------------------------------------
try:
    morphComp=np.load(resultsDir+r'\morphComp.npy')
    props=np.load(resultsDir+r'\props.npy')
except:
    print("Error, no fibrec, morphComp, props found")

parentDir=r'C:\Users\t97721hr\Dropbox (The University of Manchester)\Fibril Tracking Algorithm\abc_january_BC_0_2'
frs= glob.glob( parentDir+r'\fibrec_rank_' + '*.npy')
resultsDir=parentDir+r'\results'

md.create_Directory(resultsDir)

rank=4
fib_rec_0=np.load(frs[rank])

nfibs_0,nplanes=fib_rec_0.shape
npix=morphComp.shape[2]
pxsize, dz=np.genfromtxt( md.find_3V_data(whichdata)+'pxsize.csv', delimiter=',')[1] #import voxel size
junk=np.nonzero(np.all(fib_rec_0==-1, axis=0))[0]

#----------------------------------------------------------------------------
#....................SELECTING FIBRILS OF A CERTAIN LENGTH....................
#-------------------------------------------------------------------------------
#Q: How long are all the fibrils in the original fibril rec?
nexist_0=np.zeros(nfibs_0, dtype='int')
for i in range(nfibs_0):
    nexist_0[i]=np.max(np.nonzero(fib_rec_0[i]>-1))-np.min(np.nonzero(fib_rec_0[i]>-1))+1
longfibs=np.where(nexist_0>nplanes*desired_length)[0]  #the indices of the long fibirls
title_='Number of entries %i, Number above 90pc %i, Percentage %.1f '%(nfibs_0, longfibs.size, 100*longfibs.size/nfibs_0)

md.my_histogram(100*nexist_0/nplanes,'Number of slices present',title=title_, binwidth=5)

#Erasing fibril record for short fibrils. The only way to map between the two is using longfibs. Reindexing also!
fib_rec=fib_rec_0[longfibs]
nexist=nexist_0[longfibs]
nfibs=longfibs.size
lengths_scaled=np.zeros(nfibs)  #worm like length / n planes present
direction_unit_vectors=np.zeros((nfibs, 3))

#Q: how many fibs are we capturing in cross-section?
meanperplane=np.mean(np.apply_along_axis(np.max, 1, np.reshape(morphComp, (nplanes,npix**2 ))))
print('fraction captured in cross section', nfibs/meanperplane)


#%%---------------------------------------------------------------------------
#..............................ANIMATIONS, OPTIONAL....................
#-------------------------------------------------------------------------------
# DROPPED
md.export_animation(resultsDir,r'\rank_' +str(rank)+ '_dropped_fibril_inquiry_50to90', morphComp,half_length_fibril_indices,fib_rec_0, dt=1000)

#%% ALL
#md.animation_inline(morphComp,np.arange(nfibs), fib_rec,0,2)
md.export_animation(resultsDir,r'\rank_' +str(rank)+ '_90pc_plus_animation', morphComp,np.arange(nfibs),fib_rec, dt=1000)

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
            coOrds_2D.append((props[pID, i, 0:2]))

        return np.append(pxsize*np.mean(np.array(coOrds_2D), axis=0), dz*pID)
def fibCoords(i):
    """
    calculates the coordinates in 3d real space of a fibril (i).
    """
    co_ords=np.full((nplanes, 3),-1.)  #an array of the centroid co-ordinates for each fibril
    for pID in range(nplanes):
        if fib_rec[i, pID]!=-1:
            co_ords[pID, 0:2]=(props[pID, fib_rec[i,pID], 0:2])*pxsize
            co_ords[pID, 2]=pID*dz
    co_ords=co_ords[co_ords[:,2]>-1.]  #getting rid of junk slices / places where absent
    #This stuff is to draw a line of best fit! Leaving it out
    #mean = np.mean(co_ords, axis=0)
    #uu, dd, vv=np.linalg.svd(co_ords-mean)
    #direction=vv[0]
    return co_ords #, mean, direction
def plotfibril_withLOBF(i):
    co_ords, mean, direction=calculaterr_coordinates(i)
    linepts =0.5*np.linalg.norm(co_ords[0]-co_ords[-1])* direction *np.mgrid[-1:1:2j][:, np.newaxis]
    linepts += mean # shift by the mean to get the line in the right place
    ax = m3d.Axes3D(plt.figure())
    ax.scatter3D(*co_ords.T)
    ax.plot3D(*linepts.T)
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y',fontsize=20,)
    ax.set_zlabel('z', fontsize=20 )
    ax.view_init(elev=50, azim=160)
    plt.show()
def coOrds_to_length(co_ords):
    """
    input an Nx3 numpy array representing a list of 3D coordinates, and it will calculate the length of the 'worm like' length joining all the co-ordinates in 3d
    """
    L=0
    for j in range((co_ords.shape[0])-1): #j steps through planes in co-ords
        dr=co_ords[j]-co_ords[j+1]
        L+=np.linalg.norm(dr)
    return L

#Calculate fascicle length
fas_coord_list=[]
for pID in range (nplanes-1):
    if np.all(fascicleCoord(pID))!=None:
        fas_coord_list.append(fascicleCoord(pID))
fas_len=coOrds_to_length(np.array(fas_coord_list))

#Calculate length of each fibril
for i in range (nfibs):
    lengths_scaled[i]=coOrds_to_length(fibCoords(i))
lengths_scaled*=nplanes/(fas_len*nexist)

#How long are the long fibrils?
md.my_histogram((lengths_scaled-1)*100, 'Critical Strain (%)', title='', binwidth=.5,filename=resultsDir+r'\rank_' +str(rank)+'_CS_dist.png')
np.save(resultsDir+r'\rank_' +str(rank)+r'\scaledlengths', lengths_scaled)

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
np.save(resultsDir+r'\rank_' +str(rank)+ '_fib_MFDs', fib_MFDs)
md.my_histogram(fib_MFDs, 'Minimum Feret Diameter (nm)', 'Minimum Feret Diameter distribution', filename=resultsDir+r'\rank_' +str(rank)+'_MFD_dist.png')

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
np.save(resultsDir+r'\rank_' +str(rank) +'_area.npy', fibrilArea)
md.my_histogram(fibrilArea/100, 'Area ($10^3$ nm$^2$)', 'Cross Sectional Area of tracked fibrils', binwidth=50)
#%%----------------Length vs cross secitonal Area

plt.plot(lengths_scaled,fibrilArea/10**6, '.r')
plt.xlabel('Normalised lengths')
plt.ylabel('Cross Sectional Area (um$^2$)')
plt.show()

#%%----------------------------------------------------------------------------
#....................TESTING FOR STATISTICAL SIGNIFICANCE ....................
#------------------------------------------------------------------------------

seg_MFDs=np.ravel(props[:,:,5]*pxsize) #MFDs of all the segments in the volume
lower, upper=(80, 300)
relevantSegMFDs=seg_MFDs[(seg_MFDs>lower) & (seg_MFDs<upper)]
kstest=stats.ks_2samp(fib_MFDs, relevantSegMFDs)
result="reject" if kstest[1]<0.05 else "accept"

md.my_histogram([fib_MFDs, relevantSegMFDs], 'Feret Diameter (nm)', title=f'$H_0$, these two samples come from the same distribution \n p={kstest[1]:.2e}: {result}', labels=['Fibrils', 'Segments'], dens=True, binwidth=20, cols=['red', 'lime'], filename=resultsDir+r'\rank_' +str(rank)+'_statistical_significance_CS_dist.png')
x=np.linspace(upper, lower, 1000)
#plt.plot(np.linspace(upper, lower, 1000), stats.kde.gaussian_kde(relevantSegMFDs)(x))


#%%----------------------------------------------------------------------------
#....................DROPPED FIBRIL INQUIRIES....................
#-------------------------------------------------------------------------------
#Q: which ones are nearly full length but not quite 6/11/2020
half_length_fibril_indices=np.nonzero((0.5*nplanes<nexist_0)&(nexist_0<=desired_length*nplanes))[0]
def plot_half_length():
    plt.plot(np.count_nonzero(fib_rec_0[half_length_fibril_indices]>-1, axis=0), '.-b')
    plt.xlabel("Plane (/100)")
    plt.ylabel("Nfibrils")
    plt.grid()
    plt.xticks(np.arange(0,nplanes, 10))
    plt.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    plt.title("Examining the fibrils which appear in 50-90% of all planes")
    plt.show()
plot_half_length()
##%% Q: where are the fibril ends in the half length group?
def plot_fib_ends_half_length():
    fibril_ends=[]
    for fID in half_length_fibril_indices:
        temp=np.nonzero(fib_rec_0[fID]>-1)[0]
        fibril_ends.extend([temp[0],temp[-1]])
    fibril_ends=np.array(fibril_ends)
    N=fibril_ends.size
    fibril_ends=fibril_ends[fibril_ends>0]
    fibril_ends=fibril_ends[fibril_ends<100]
    n, bins, patches = plt.hist(fibril_ends, nplanes, density=False, facecolor='g', alpha=0.75)
    plt.xlabel("Plane")
    plt.ylabel('Number')
    unique, counts = np.unique(fibril_ends, return_counts=True)
    plt.ylim(0,max(counts))
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
    n, bins, patches = plt.hist([f_tops,f_bottoms], nplanes//2, density=False, histtype='bar',color=cols,label=['top', 'tail'])
    #n, bins, patches = plt.hist(f_bottoms, nplanes, density=False, facecolor='b', alpha=0.5)
    plt.xlabel("Plane")
    plt.ylabel('Number')
    plt.title('Where are the fibril ends?')
    unique, counts = np.unique([f_tops,f_bottoms], return_counts=True)
    plt.ylim(0,max(counts))
    plt.vlines(junk+0.5, 0, max(counts), colors='black')
    plt.xticks(np.arange(0, nplanes, 10))
    plt.grid(True)
    plt.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    plt.legend()
    plt.show()
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
plot_strand_size_vs_MFD()
