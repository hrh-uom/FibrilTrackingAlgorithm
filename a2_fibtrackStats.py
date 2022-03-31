from a0_initialise import *
import numpy as np
import matplotlib.pyplot as plt
import customFunctions as md
from scipy import stats
import glob , os
import pandas as pd
from scipy.optimize import curve_fit
plt.style.use('./mystyle.mplstyle')

#---------------------USER INPUT--------------------------------------------

dirOutputs=d.dirOutputs
abc=False
print("a2:Fibtrack stats")

#-----------.IMPORT DATA FROM FIBRIL MAPPING---------------------------------
try:
    FR_0=np.load(dirOutputs+'fib_rec.npy') #original, import fibril record
    MC=np.load(dirOutputs+'morphComp.npy')
    props=np.load(dirOutputs+'props.npy')
except:
    print("Error, no fibrec, MC, props found")
if abc:
    FR_0=np.load('/Users/user/Dropbox (The University of Manchester)/fibril-tracking/nuts-and-bolts/csf-output/abc-dec21/rank0/fibrec_rank_0_a_1.00_b_2.70_c_2.85.npy')
    dirOutputs='/Users/user/Dropbox (The University of Manchester)/fibril-tracking/nuts-and-bolts/csf-output/abc-dec21/rank0/'
md.create_Directory(dirOutputs+'stats')

#-~~~~~~~~~~~~TRIM FIBRIL RECORD IF NEEDED ~~~~~~~~~~~~
if os.path.isfile(dirOutputs+f'fib_rec_trim_{d.frac*100}.npy'):
    print("Loading")
    FR=np.load(dirOutputs+f'fib_rec_trim_{d.frac*100}.npy') #original, import fibril record
else:
    print("Trimming")
    FR=md.trim_fib_rec(FR_0, MC, dirOutputs, d.frac)
nF, nP=FR.shape
#%%--------------FASCICLE LENGTHS---------------------------------

def fascicleCoord(pID):
    """
    Calculates the mean co-ordinate of all fibrils at slice pID
    """
    objects_in_plane=FR[:, pID][ FR[:, pID]>-1]
    if objects_in_plane.size!=0: #ignoring d.junk slices
        coOrds_2D=[]
        for i in objects_in_plane:
            coOrds_2D.append((props[pID, i, 0:2]*d.pxsize))
        return np.append(np.mean(np.array(coOrds_2D), axis=0), d.dz*pID)
    else:
        return np.array([-1,-1, -1])
def fibCoords(i):
    """
    calculates the coordinates in 3d real space of a fibril (i).
    """
    co_ords=np.full((nP, 3),-1.)  #an array of the centroid co-ordinates for each fibril
    for pID in range(nP):
        if FR[i, pID]!=-1:
            co_ords[pID, 0:2]=(props[pID, FR[i,pID], 0:2])*d.pxsize
            co_ords[pID, 2]=pID*d.dz
    co_ords_trim=co_ords[co_ords[:,2]>-1.]  #getting rid of d.junk slices / places where absent
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

def calculate_fascicle_length():
    """
    Calculate the arclength of the fascile
    """
    fas_coord_list=[]
    for pID in range (nP-1):
        if np.isin(pID, d.junk)==False:
            fas_coord_list.append(fascicleCoord(pID))
    fas_len=coOrds_to_length(np.array(fas_coord_list))
    return fas_len, fas_coord_list
fas_len, fas_coord_list=calculate_fascicle_length()


#%% WHERE IS THE FASCICLE GOING

def fascicle_travel():
    import matplotlib.cm as cm
    from matplotlib.collections import LineCollection


    fig, ax=plt.subplots(figsize=(10,10))

    fas_arr=np.array(fas_coord_list)
    x=fas_arr[:,0]/d.pxsize
    y=fas_arr[:,1]/d.pxsize
    cols = np.linspace(0,100*(1+nP//100),len(x))

    ax.set_aspect(1)
    # ax.set_xlim(450,550);ax.set_ylim(420, 520)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    ax.set_xlabel('x (pixels)');ax.set_ylabel('y (pixels)')
    lc = LineCollection(segments, cmap='gist_rainbow')
    lc.set_array(cols); lc.set_linewidth(2)
    line = ax.add_collection(lc)
    fig.colorbar(line,fraction=0.046, pad=0.04)

    plt.savefig(dirOutputs+'stats/fascicle-travel.png',bbox_inches='tight' ,pad_inches = 0);plt.show()
fascicle_travel()

#%%------------------PDFs for fitting----------------------

def lognorm_pdf(x, s, u):
    """
    Log normal distribution
    """
    A   =   1                       /       (x*s*np.sqrt(2*np.pi))
    B   =   (np.log(x)-u)**2       /       (2*s**2)
    return A * np.exp(-B)

def normal_pdf(x, s, u):
    """
    Normal distribution
    """
    A   =   1       /          (s * np.sqrt(2*np.pi))
    B   =   0.5     *          ((x-u)/s)**2
    return A * np.exp (-B)

def bi_pdf(x, s1, u1, s2, u2):
    """
    bimodal Normal distribution
    """
    return 0.5*(normal_pdf(x, s1, u1) + normal_pdf(x, s2, u2))

def tri_pdf(x, s1, u1, s2, u2, s3, u3):
    """
    trimodal Normal distribution
    """
    return (normal_pdf(x, s1, u1) + normal_pdf(x, s2, u2)+normal_pdf(x, s3, u3))/3


#%%------------------Fibril Length

def calculate_fibril_lengths():
    #Calculate length of each fibril
    #Q: How long are all the fibrils in the fibril rec?
    nexist=np.zeros(nF, dtype='int')
    for i in range(nF):
        nexist[i]=np.max(np.nonzero(FR[i]>-1))-np.min(np.nonzero(FR[i]>-1))+1


    lens=np.zeros(nF)  #worm like length / n planes present
    for i in range (nF):
        lens[i]=coOrds_to_length(fibCoords(i)[0])
    lens*=nP/(fas_len*nexist)
    return lens, nexist

lens, nexist=calculate_fibril_lengths()

def plot_fibril_lengths(lens):

    leny, lenx=np.histogram(lens, bins=50, density=True)
    lenx=(lenx+np.diff(lenx)[0]/2)[: -1]
    pars, cov=curve_fit(normal_pdf, lenx, leny)
    x=np.linspace(np.min(lens), np.max(lens), 1000)
    fit=[x, normal_pdf(x, pars[0], pars[1])]
    print(f'Critical Strain. Fibril strands appear in {d.l_min/1000}um of z distance')
    print(f'Critical strain mean {np.mean(lens)}, sd {np.std(lens)}')
    md.my_histogram((lens), 'Length relative to fascicle', binwidth=.005,filename=dirOutputs+f'stats/CS_dist_{d.l_min}nm_{d.frac*100}.png', dens=False, fitdata=fit, fitparams=pars)
    np.save(dirOutputs+f'stats/scaledlengths_{d.l_min}nm_{d.frac*100}', lens)

plot_fibril_lengths(lens)

#%%---------------------------Feret Diameter of each fibril
def calculate_MFDs():
    def fibril_MFD(i, FR): #maps between props and fibrec
        feret_planewise=np.full(nP,-1.)  #an array of the centroid co-ordinates for each fibril
        for pID in range(nP):
            if FR[i, pID]!=-1:
                feret_planewise[pID]=(props[pID, FR[i,pID], 5])*d.pxsize
        feret_planewise=feret_planewise[feret_planewise>-1.]  #getting rid of d.junk slices / places where absent
        mean = np.mean(feret_planewise, axis=0)
        return mean,feret_planewise

    mfds= np.array([fibril_MFD(i, FR)[0] for i in range(nF)])
    np.save(dirOutputs+f'mfds_{d.l_min}nm_{d.frac*100}', mfds)
    return mfds

mfds=calculate_MFDs()


def plot_mfds():
    mfdy, mfdx=np.histogram(mfds, bins=30, density=True)
    mfdx=(mfdx+np.diff(mfdx)[0]/2)[: -1]
    pars, cov=curve_fit(bi_pdf, mfdx, mfdy, bounds=([0, 75, 0, 200],[np.inf, 175, np.inf, 250]))
    xx=np.linspace(np.min(mfds), np.max(mfds), 1000)
    # fit=np.ndarray.flatten(np.array([xx, normal_pdf(xx,pars[0], pars[1])]
    fit=np.vstack([np.concatenate([xx,xx]),np.concatenate([normal_pdf(xx,pars[0], pars[1]), normal_pdf(xx,pars[2], pars[3])])])

    md.my_histogram(mfds, 'Minimum Feret Diameter (nm)', dens=False,filename=dirOutputs+f'stats/MFD_dist_{d.l_min}nm_{d.frac*100}.png', fitdata=False)

plot_mfds()

#%%----------------Orientation
def calculate_orientation():
    fas_coords=np.array([fascicleCoord(pID) for pID in range (nP)]) #This includes d.junk planes
    anglelis=[]
    for fID in range(nF):
        fib_coords_values=fibCoords(fID)[1]
        for pID in np.arange(nP-1):
            if np.all(np.stack((fib_coords_values[pID], fib_coords_values[pID+1]))>=0):
                a=fas_coords[pID+1]-fas_coords[pID]
                b=fib_coords_values[pID+1]-fib_coords_values[pID]
                angle=(180/(2*np.pi))*np.arccos(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))
                if angle>180:
                    angle=angle-180
                anglelis.append(angle)
                # print(f'fid {fID} pID {pID}')
    print(f'mean {np.mean(np.array(anglelis))}, sd {np.std(np.array(anglelis))}')
    return np.array(anglelis)

oris=calculate_orientation()

oriy, orix=np.histogram(oris, bins=50, density=True)
orix=(orix+np.diff(orix)[0]/2)[: -1]
pars, cov=curve_fit(lognorm_pdf, orix, oriy)
xx=np.linspace(0.01, 60, 2000)
fit=[xx, lognorm_pdf(xx, pars[0], pars[1])]

md.my_histogram(oris, 'Direction relative to mean (degrees)',  dens=True,binwidth=2, pi=False,filename=dirOutputs+'/stats/orientation.png', fitdata=fit, fitparams=pars)


#%% ------------------------Area of each fibrils

def fibril_area(i):
    """
    Delivers fibril area in nm^2, for some fibril in the Fibril Record i
    """
    area_planewise=np.full(nP,-1.)  #an array of the areas for each fibril
    for pID in range(nP):
        if FR[i, pID]!=-1:
            area_planewise[pID]=(props[pID, FR[i,pID],3])*(d.pxsize**2)
    area_planewise=area_planewise[area_planewise>-1.]  #getting rid of d.junk slices / places where absent
    mean = np.mean(area_planewise)
    return mean, area_planewise
fibrilArea=np.array([fibril_area(i)[0] for i in range(nF)])
np.save(dirOutputs+f'area_{d.l_min}nm_{d.frac*100}.npy', fibrilArea)
md.my_histogram(fibrilArea/100, 'Area ($10^3$ nm$^2$)', 'Cross Sectional Area of tracked fibrils', binwidth=50)



#%%----------------Length vs cross secitonal Area

plt.plot(lens,fibrilArea/10**6, '.r')
plt.xlabel('Normalised lengths')
plt.ylabel('Cross Sectional Area (um$^2$)')
plt.show()

#%%WHAT ARE THE REALLY LONG FIBRILS UP to
fig, (ax1, ax2)=plt.subplots(1,2)
XLfibs=np.where(lens>1.1)[0]
ax1.set_xlabel('Normalised lengths')
ax1.set_ylabel('Minimum Feret Diameter (nm)')
ax1.plot(lens[XLfibs],mfds[XLfibs], '.r', )
ax2.set_xlabel('Minimum Feret Diameter (nm)')
ax2.set_ylabel('Density')
ax2.hist(mfds[XLfibs], density=True)
fig.tight_layout()
plt.savefig(dirOutputs+'stats/xl-fibs.png')
plt.show()
xl_fibs_render=False
if xl_fibs_render:
    labels=md.label_volume(MC, XLfibs, FR, 695)
    md.export_animation(dirOutputs, XLfibs, labels, title='stats/xl-fibrils-animation', step=5)
    md.volume_render(labels, 0, nP, 0, 1000,d.pxsize, d.dz,dirOutputs,'stats/XLvolumerender')
#%% d1 fibrils
def d1_3d():
    d1_fibrils=np.where(mfds<100)[0]
    dirOutputs
    labels=md.label_volume(MC, d1_fibrils, FR, 695)
    md.export_animation(dirOutputs, d1_fibrils, labels, title='stats/D1-animation', step=5)
    md.volume_render(labels, 0, nP, 0, 1000,d.pxsize, d.dz,dirOutputs,'stats/D1-render')
#%%------TESTING FOR STATISTICAL SIGNIFICANCE ----------------------------------
#Q: which segments don't get picked up?
def statistical_significance():
    planes_no_cells=np.array([1,25,47,85,95,96,111, 129,155,158,277,291, 296])-1 #read off imageJ
    tracked_FD=np.ones([1]);untracked_FD=np.ones([1]);

    # for pID in planes_no_cells:
    #     tracked=np.setdiff1d(np.unique(FR[:,pID]),np.array([-1]))
    #     tracked_FD=np.concatenate((tracked_FD,props[pID, tracked, 5]*d.pxsize))
    #     untracked=np.setdiff1d(np.arange(np.count_nonzero(props[0,:,3])),tracked)
    #     untracked_FD=np.concatenate((untracked_FD,props[pID, untracked, 5]*d.pxsize))
    for pID in [nP//2]:
        trackedIDs=np.unique(FR[:,pID])[1:]
        tracked_FDs=props[pID,trackedIDs, 5]*d.pxsize
        allIDs=np.unique(MC[pID]-1)[1:]
        allFDs=props[pID,allIDs, 5]*d.pxsize


    lower, upper=90,350
    rel_all_FD=allFDs[(allFDs>lower) & (allFDs<upper)]
    rel_tracked_FD=tracked_FDs[(tracked_FDs>lower) & (tracked_FDs<upper)]

    kstest=stats.ks_2samp(rel_tracked_FD, rel_all_FD)
    result="reject" if kstest[1]<0.05 else "accept"
    title=f'$H_0$, these two samples come from the same distribution. p={kstest[1]:.2f}: {result}\n Distribution limited to ({lower}, {upper}) nm'
    print(title)

    # md.my_histogram([rel_tracked_FD, rel_all_FD],'Feret diameter (nm)', binwidth=20,cols=['red', 'lime'], dens=False, labels=['Tracked fibrils', 'All segments'],filename=dirOutputs+f'stats/statistical_significance_CS_dist_{d.l_min}nm_{d.frac*100}.png', leg=True)

    fig,ax=plt.subplots(figsize=(10, 7))
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
    ax2.set_ylabel('fraction of segments captured')

    filename=dirOutputs+f'stats/statistical_significance_CS_dist_{d.frac}.png'
    plt.savefig(filename);
    plt.show()

statistical_significance()
#%%---------------DROPPED FIBRIL INQUIRIES--------------------------------------
#Q: read original fibril record, before chopping.
nF_0,nP=FR_0.shape
_, _, npix=MC.shape
#Q: how many fibs are we capturing in cross-section?
meanperplane=np.mean(np.apply_along_axis(np.max, 1, np.reshape(MC, (nP,npix**2 ))))
n=0
for i in range(nP):
    n+=FR[i][FR[i]>-1].size


print('fraction captured in cross section', (n/nP)/meanperplane)


#Q: How long are all the fibrils in the original fibril rec?
nexist_0=np.zeros(nF_0, dtype='int')
for i in range(nF_0):
    nexist_0[i]=np.max(np.nonzero(FR_0[i]>-1))-np.min(np.nonzero(FR_0[i]>-1))+1


title_=f'Strands: {nF_0}, > {d.l_min}nm in z: {nF} $\sim$ {100*nF/nF_0:.0f}%'
print(title_)
md.my_histogram(nexist_0/nP,'fraction of planes present', binwidth=.05, filename=dirOutputs+'stats/planes_present')
md.my_histogram(nexist/nP,'fraction of planes present', binwidth=.05, filename=dirOutputs+'stats/planes_present_zoom')
#%%     JUNK PLANES QUERIES

def plot_fib_tops_bottoms():

    f_tops=[];f_bottoms=[]
    for fID in range(nF):
        temp=np.nonzero(FR[fID]>-1)[0]
        f_tops.append(temp[0])
        f_bottoms.append(temp[-1])
    cols=['red', 'lime']

    h=np.histogram(f_tops+f_bottoms, nP//10)
    plt.plot( h[1][0:-1], h[0])
    plt.ylim(0, 1.05*max(h[0]))
    plt.xlabel("Plane")
    plt.ylabel('Number')

    for i in range(d.junk.shape[0]):
        plt.arrow(d.junk[i],350,    0, -200,  lw=1, length_includes_head=True, head_width=10)

    plt.savefig(dirOutputs+'stats/fibrilends');plt.show()
plot_fib_tops_bottoms()
