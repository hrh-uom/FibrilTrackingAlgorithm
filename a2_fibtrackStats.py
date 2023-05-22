import numpy as np
import matplotlib.pyplot as plt
import customFunctions as md
import a0_initialise as a0
from scipy import stats
import glob , os
import pandas as pd
from scipy.stats import norm, skewnorm
import matplotlib.patches as mpatches
from  matplotlib.colors import ListedColormap
from tqdm import tqdm
from matplotlib import animation
from scipy.optimize import curve_fit

#---------------------USER INPUT----------------.----------------------------
print(f'a2:Fibtrack stats {md.t_d_stamp()}')
atom=False
d, MC, props = a0.initialise_dataset()

#-----------.IMPORT DATA FROM FIBRIL MAPPING---------------------------------
def import_from_FTA():
    try:
        dirOutputs=d.dirOutputs
        FR_0=np.load(dirOutputs+'fib_rec.npy') #original, import fibril record
    except:
        print("Error, no fibrec found")
    #-~~~~~~~~~~~~TRIM FIBRIL RECORD IF NEEDED ~~~~~~~~~~~~
    try:
        FR=np.load(glob.glob(dirOutputs+f'fib_rec_trim*')[0])
        print("Loading trimmed FR")
    except:
        print("Trimming")
        FR=md.trim_fib_rec(FR_0, MC, dirOutputs, d.frac)
    md.create_Directory(dirOutputs+'stats')
    nF, _=FR.shape
    return dirOutputs, FR , FR_0, nF
dirOutputs, FR, FR_0, nF = import_from_FTA()
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
    calculates the coordinates in 3d real space of a fibril (i) in nm - from the pizel sizes
    """
    co_ords=np.full((d.nP, 3),-1.)  #an array of the centroid co-ordinates for each fibril
    for pID in range(d.nP):
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
    fas_coord=[]
    for pID in range (d.nP):
        if np.isin(pID, d.junk)==False:
            fas_coord.append(fascicleCoord(pID))
    fas_len=coOrds_to_length(np.array(fas_coord))
    return fas_len, np.array(fas_coord)
fas_len, fas_coord=calculate_fascicle_length()

print (f"{d.dataset} the length of the fascicle is {fas_len}")
#%% WHERE IS THE FASCICLE GOING
def fascicle_travel():
    import matplotlib.cm as cm
    from matplotlib.collections import LineCollection
    fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(20,6))
    x=fas_coord[:,0]/d.pxsize
    y=fas_coord[:,1]/d.pxsize
    cols = np.linspace(0,100*(1+d.nP//100),len(x))

    # ax.set_aspect(1)
    ax1.set_xlim(500,1000);ax1.set_ylim(200, 1000)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    ax1.set_xlabel('x (pixels)');ax1.set_ylabel('y (pixels)')
    lc = LineCollection(segments, cmap='gist_rainbow')
    lc.set_array(cols); lc.set_linewidth(2)
    line = ax1.add_collection(lc)
    fig.colorbar(line,ax=ax1, fraction=0.046, pad=0.04)

    z=fas_coord[:,2]
    fas_loc_2D=np.linalg.norm(fas_coord[:, 0:2], axis=1)/1000

    ax2.plot(z/d.dz,fas_loc_2D)
    ax2.set_xlabel('Plane'); ax2.set_ylabel('Fascicle Movement')
    plt.savefig(dirOutputs+'stats/fascicle-travel.png');

# fascicle_travel()

#%%------------------Fibril Length
def calculate_fibril_lengths():
    #Calculate length of each fibril
    #Q: How long are all the fibrils in the fibril rec?
    nexist=np.zeros(nF, dtype='int')
    for i in range(nF):
        nexist[i]=np.max(np.nonzero(FR[i]>-1))-np.min(np.nonzero(FR[i]>-1))+1
    lens=np.zeros(nF)  #worm like length
    criticalstrains=np.zeros(nF)  #worm like length / fascicle length for that fibril
    fas_coord_inc_junk=np.array([fascicleCoord(j) for j in range(d.nP)])
    for i in tqdm(range (nF)):
        fib_exist_in=np.argwhere(fibCoords(i)[1][:,0]>0 ).T [0]#Indices of where fibril exists
        faslen_rel=coOrds_to_length(fas_coord_inc_junk[fib_exist_in])
        criticalstrains[i]=coOrds_to_length(fibCoords(i)[0])/faslen_rel
        lens[i]=coOrds_to_length(fibCoords(i)[0])
    np.save(dirOutputs+f'stats/criticalstrains', criticalstrains)
    return criticalstrains, nexist, lens
criticalstrains,nexist, lens=calculate_fibril_lengths()


#%%CSD AND LENGTHS

def plot_critical_strain(criticalstrains, fitting=False):
    leny, lenx=np.histogram((criticalstrains), bins=50, density=True)
    lenx=(lenx+np.diff(lenx)[0]/2)[: -1]
    pars, cov=curve_fit(md.normal_pdf, lenx, leny)
    x=np.linspace(np.min(criticalstrains), np.max(criticalstrains), 1000)
    fit=[x, md.normal_pdf(x, pars[0], pars[1])]
    print(f'Critical Strain. Fibril strands appear in {d.l_min/1000}um of z distance')
    print(f'Critical strain mean {np.mean(criticalstrains)}, sd {np.std(criticalstrains)}')

    if fitting:
        md.my_histogram((criticalstrains), 'Length relative to fascicle',atom, binwidth=.05,filename=dirOutputs+f'stats/CS_dist_fit.png', dens=False, fitdata=fit, fitparams=pars)
    else:
        md.my_histogram((criticalstrains), 'Length relative to fascicle',atom,show=False,  binwidth=.05,filename=dirOutputs+f'stats/CS_dist.png', dens=False)

def plot_scaled_lens(lens, nexist):
    scaledlengths=lens*(d.nP/nexist)/1000 #In um
    md.my_histogram((scaledlengths), 'Fibril length ($\mu$m)',atom, binwidth=0.2,filename=dirOutputs+f'stats/scaledlengths.png', dens=False)
    np.save(dirOutputs+f'stats/scaledlengths', scaledlengths)
# plot_scaled_lens(lens, nexist)
plot_critical_strain(criticalstrains, fitting=True)
# plot_scaled_lens(lens, nexist)

#%%HELICES
# i=10
def helix_right_left_test(i, plot_it=True, saveit=False, rank=0):
    """
    i = fibril number
    returns 1 if fibril arc is clockwise in xy plane -> right handed helix in 3D
    https://stackoverflow.com/questions/25252664/determine-whether-the-direction-of-a-line-segment-is-clockwise-or-anti-clockwise
    https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
    """

    def savitzky_golay(y, window_size, order, deriv=0, rate=1):
        """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters
        ----------
        y : array_like, shape (N,)
            the values of the time history of the signal.
        window_size : int
            the length of the window. Must be an odd integer number.
        order : int
            the order of the polynomial used in the filtering.
            Must be less then `window_size` - 1.
        deriv: int
            the order of the derivative to compute (default = 0 means only smoothing)
        Returns
        -------
        ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.
        Examples
        --------
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        #plt.show()
        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
           Data by Simplified Least Squares Procedures. Analytical
           Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
           W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
           Cambridge University Press ISBN-13: 9780521880688
        """
        from math import factorial

        try:
            window_size = np.abs(int(window_size))
            order = np.abs(int(order))
        except:# ValueError, msg:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve( m[::-1], y, mode='valid')
    C=fibCoords(i)[0].T
    #Smooth fibril arc
    x = savitzky_golay(C[0], 61, 3) # window size 51, polynomial order 3
    y = savitzky_golay(C[1], 61, 3) # window size 51, polynomial order 3

    sum=0
    total=0
    for i in range (x.size-2):
        # i=3
        vecU=np.array([x[i+1]-x[i] ,(y[i+1]-y[i])]) #finds cross product of two consecutive segments of path length
        vecV=np.array([x[i+2]-x[i+1] ,(y[i+2]-y[i+1])])
        total= total+1 if np.cross(vecU, vecV) > 0 else total-1
        #if more are negative we say the curve is clockwise, and vise versa
    H_value=total/(x.size-2) #flipped the sign so that left handed = negative because it looks better in the plot
    if plot_it:
        print ('Clockwise' if total <0 else 'Anticlockwise')
        fig, ax=plt.subplots(figsize=(6, 4), dpi=100)
        ax.plot(C[0], C[1])
        ax.set_title(f'Fibril number {i} helicity {H_value:.4f}')
        ax.plot(x, y)
        ax.plot([C[0,0]], [C[1,0]], 'ro')
        if saveit:
            plt.savefig(dirOutputs+f'stats/chiral/helix_{rank}_fib_{i}')
        #plt.show()

        plt.close()
    return H_value
helix_arr=np.zeros(nF)
for j in tqdm(range(nF)):
    helix_arr[j]=helix_right_left_test(j, plot_it=False)
#%%

print(helix_arr)


def export_fib_path_imgs():
    helix_sort_fibs=np.argsort(helix_arr) #list of firbils in helical order

    for j in tqdm(range(len(helix_sort_fibs))):
        i=helix_sort_fibs[j]
        helix_right_left_test(i, plot_it=True, saveit=True, rank=j)

def helicity_plot(cutoff=0, xmin=-1, xmax=1):
    nF=FR.shape[0]
    fig, ax=plt.subplots()
    ax2=ax.twinx()
    bins_=np.linspace(xmin, xmax, 40)
    N, bins, patches =     ax.hist(helix_arr, bins=bins_, edgecolor='k', density=False)
    ax.set_ylim(0, max(N)+10)

    skew, mean,std=skewnorm.fit(helix_arr)
    xmin, xmax = -0.5, 0.5
    x = np.linspace(xmin, xmax, 100)
    y = skewnorm.pdf(x, skew, mean, std)


    ax2.plot(x, y, '--k', label='Normal PDF')
    ax2.set_ylim(0, max(y))
    ax2.grid(False)
    ax2.set_ylabel('Probability density')

    textstr=f'$\mu$ = {mean:.3f}\n$\sigma$ = {std:.3f}\n$a$ = {skew:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.1, 0.9, textstr, transform=ax.transAxes, fontsize=18,
    verticalalignment='top', bbox=props)

    RH_helical_fibs=np.where((helix_arr)<-cutoff)[0]
    R=LH_helical_fibs=np.where((helix_arr)>cutoff)[0]
    which_helical=np.concatenate((LH_helical_fibs, RH_helical_fibs), axis=None)
    nL=len(LH_helical_fibs); nR=len(RH_helical_fibs) ; nH=nL+nR

    for bar in patches:
        if (bar.get_x() < -cutoff):
            bar.set_facecolor("r")
        if (bar.get_x() > cutoff):
            bar.set_facecolor("g")


    right = mpatches.Patch(color='red', label=f'Right-handed\n$n=${nR}')
    non = mpatches.Patch(color='b', label=f'Non-helical \n$n=${nF-nH}')
    left = mpatches.Patch(color='g', label=f'Left-handed\n$n=${nL}')

    plt.legend(handles=[right,non,left],loc=1,fontsize=18)

    ax.set_xlabel('Chirality') ; ax.set_ylabel('Number')
    plt.savefig(dirOutputs+f'stats/chirality')
    # #plt.show()
    plt.close()
    return which_helical
hel_fibs=helicity_plot()

nonhel_fibs=np.setdiff1d(np.arange(0, nF), hel_fibs)

#%%Tortuosity

tort=np.zeros(lens.shape)
for i in tqdm(range(nF)):
    first=fibCoords(i)[0][0]
    last=fibCoords(i)[0][-1]
    end_to_end=np.linalg.norm(last-first)
    tort[i]=100*(lens[i]/end_to_end - 1)

x=np.linspace(np.min(tort), np.max(tort), 1000)
leny, lenx=np.histogram((tort), bins=50, density=True)
lenx=(lenx+np.diff(lenx)[0]/2)[: -1]
pars, cov=curve_fit(md.lognorm_pdf, lenx, leny)
fit=[x, md.lognorm_pdf(x, pars[0], pars[1])]
md.my_histogram(tort, binwidth=2,xlabel='Tortuosity', show=False, fitdata=fit,fitparams=pars, dens=False,filename=dirOutputs+'stats/tort' )

#%% ------------------------Area of each fibrils

def fibril_area(i):
    """
    Delivers fibril area in nm^2, (since the pixel size is given in nm) for some fibril in the Fibril Record i
    """
    area_planewise=np.full(d.nP,-1.)  #an array of the areas for each fibril
    for pID in range(d.nP):
        if FR[i, pID]!=-1:
            area_planewise[pID]=(props[pID, FR[i,pID],3])*(d.pxsize**2)
    area_planewise=area_planewise[area_planewise>-1.]  #getting rid of d.junk slices / places where absent
    mean = np.mean(area_planewise)
    return mean, area_planewise
def plot_fibril_area():
    np.save(dirOutputs+f'area_{d.l_min}nm_{d.frac*100}.npy', area)
    md.my_histogram(area/100, 'Fibril cross sectional area ($10^3$ nm$^2$)', atom, binwidth=25, filename=dirOutputs+'/stats/fibrilarea.png')
area=np.array([fibril_area(i)[0] for i in range(nF)])
plot_fibril_area()

#%% MFDS

def calculate_MFDs():
    def fibril_MFD(i, FR): #maps between props and fibrec
        """
        gives fibril MFD in nm as this is the units for the pixel size
        """
        feret_planewise=np.full(d.nP,-1.)  #an array of the centroid co-ordinates for each fibril
        for pID in range(d.nP):
            if FR[i, pID]!=-1:
                feret_planewise[pID]=(props[pID, FR[i,pID], 5])*d.pxsize
        feret_planewise=feret_planewise[feret_planewise>-1.]  #getting rid of d.junk slices / places where absent
        mean = np.mean(feret_planewise, axis=0)
        return mean,feret_planewise

    mfds= np.array([fibril_MFD(i, FR)[0] for i in range(nF)])
    np.save(dirOutputs+f'mfds_{d.l_min}nm_{d.frac*100}', mfds)
    return mfds
def plot_mfds(bi=True, tri=False, fitting=False):
    mfdy, mfdx=np.histogram(mfds, bins=50, density=True)
    mfdx=(mfdx+np.diff(mfdx)[0]/2)[: -1]
    xx=np.linspace(np.min(mfds), np.max(mfds), 1000)

    if bi:
        u1_g=100;     s1_g=100
        u2_g=250;    s2_g=100
        pars, cov=curve_fit(md.bi_pdf, mfdx, mfdy, p0=[s1_g, u1_g,s2_g, u2_g , 0.5])
        fit=[xx, md.bi_pdf(xx,pars[0], pars[1], pars[2], pars[3], pars[4])]
    if tri:
        u1_b=[75, 120];     s1_b=[0, np.inf]
        u2_b=[125, 175];    s2_b=[0, np.inf]
        u3_b=[200, 250];    s3_b=[0, np.inf]
        # u1_b=[0, np.inf];     s1_b=[0, np.inf]
        # u2_b=[0, np.inf];    s2_b=[0, np.inf]
        # u3_b=[0, np.inf];    s3_b=[0, np.inf]

        pars, cov=curve_fit(md.tri_pdf, mfdx, mfdy, bounds=([s1_b[0], u1_b[0], s2_b[0], u2_b[0], s3_b[0], u3_b[0], 0, 0],[s1_b[1], u1_b[1], s2_b[1], u2_b[1], s3_b[1], u3_b[1],  1, 1]))
        fit=[xx, md.tri_pdf(xx,pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars [7])]

    if fitting:
        md.my_histogram(mfds, 'Minimum Feret Diameter (nm)', atom, dens=False,filename=dirOutputs+f'stats/minimum_feret_diamerer', fitdata=fit, binwidth=5,fitparams=pars)
    else:
        md.my_histogram(mfds, 'Minimum Feret Diameter (nm)', atom, dens=False,filename=dirOutputs+f'stats/MFD_dist.png',binwidth=5)

mfds=calculate_MFDs()
plot_mfds(bi=True, fitting=True)

#%%----------------Orientation
def calculate_orientation():
    fas_coords=np.array([fascicleCoord(pID) for pID in range (d.nP)]) #This includes d.junk planes
    orientation_lis=[]
    for fID in range(nF):
        anglelis=[]
        fib_coords_values=fibCoords(fID)[1]
        for pID in np.arange(d.nP-1):
            if np.all(np.stack((fib_coords_values[pID], fib_coords_values[pID+1]))>=0):
                a=fas_coords[pID+1]-fas_coords[pID]
                b=fib_coords_values[pID+1]-fib_coords_values[pID]
                angle=(180/(2*np.pi))*np.arccos(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))
                if angle>180:
                    angle=angle-180
                anglelis.append(angle)
        orientation_lis.append(np.mean(anglelis))
    return np.array(orientation_lis)
def plot_orientation(oris):
    oriy, orix=np.histogram(oris, bins=50, density=True)
    orix=(orix+np.diff(orix)[0]/2)[: -1]
    pars, cov=curve_fit(md.lognorm_pdf, orix, oriy)
    xx=np.linspace(0.01, 15, 2000)
    fit=[xx, md.lognorm_pdf(xx, pars[0], pars[1])]

    md.my_histogram(oris, f'Fibril alignment ($\degree$) ', atom, dens=False,binwidth=.5, pi=False,filename=dirOutputs+'/stats/orientation.png', fitdata=fit, fitparams=pars, units='$\degree$')

oris=calculate_orientation()
plot_orientation(oris)

#%% VOLUME FRACTION

def calculate_volume_fraction():
    goodplanes=np.setdiff1d(np.arange(d.nP), d.junk, assume_unique=False)
    vol_total=np.prod(MC[goodplanes].shape)
    VF_raw=np.count_nonzero(MC[goodplanes])/vol_total

    area_cum=0
    for fID in range(nF):
        for pID in range(d.nP):
            if FR[fID, pID]!=-1:
                area_cum+=(props[pID, FR[fID,pID],3])

    VF_FTA=area_cum/vol_total
    pd.DataFrame([[VF_raw, VF_FTA, vol_total]], columns=['VF_raw', 'VF_FTA', 'V_tot']).to_csv(dirOutputs+'stats/VF.csv')
    # return VF_raw, VF_FTA, vol_total
calculate_volume_fraction()

#%%------TESTING FOR STATISTICAL SIGNIFICANCE ----------------------------------
#Q: which segments don't get picked up?
def statistical_significance():
    planes_no_cells=np.setdiff1d(np.arange(d.nP), d.junk)
    tracked_FD=np.ones([1]);untracked_FD=np.ones([1]);
    for pID in [d.nP//2]:
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
    ax.legend(loc=2, fontsize=16)
    filename=dirOutputs+f'stats/statistical_significance_CS_dist_{d.frac}.png'
    plt.savefig(filename);
    #if atom:
        #plt.show()

statistical_significance()


#%%finding fibril ends

def object_numbers_which_end_in_plane(pID,fibrilends, tops=True):
    switch=0 if tops else 1
    which_fibs=np.where(fibrilends[:, switch]==pID)[0]
    obj_numbers=FR[which_fibs, pID]
    centroids=props[pID, obj_numbers,0:2]
    which_near_edge=np.argwhere((centroids>d.npix*0.95)|(centroids<d.npix*0.05))[:,0]
    edgeobjects=obj_numbers[which_near_edge]
    centre_objects=np.setdiff1d(obj_numbers, edgeobjects)
    return  centre_objects , edgeobjects
def find_fibril_ends():
    ends_df=pd.DataFrame(columns=['ntop_c', 'ntop_e', 'ntail_c', 'ntail_e', 'sumc', 'sume'])

    # pID=0
    nexist=np.zeros(nF, dtype='int')
    for i in range(nF):
        nexist[i]=np.max(np.nonzero(FR[i]>-1))-np.min(np.nonzero(FR[i]>-1))+1

    fibrilends=np.zeros((nF, 2), dtype='int') #first entry = top , second= bottom
    for i in range(nF):
        ends=np.nonzero(FR[i]>-1)[0][[0, -1]]
        fibrilends[i]=ends

    for pID in range(d.nP):
        ntop_c,ntop_e=map(len,object_numbers_which_end_in_plane(pID, fibrilends, tops=True))
        ntail_c,ntail_e=map(len,object_numbers_which_end_in_plane(pID, fibrilends,tops=False))

        ends_df.loc[pID]=[ntop_c, ntop_e, ntail_c, ntail_e, ntop_c+ntail_c,ntop_e+ntail_e ]
    return fibrilends, ends_df
def create_image(pID):
    top_c, top_e=object_numbers_which_end_in_plane(pID, tops=True)
    tail_c, tail_e=object_numbers_which_end_in_plane(pID, tops=False)
    im=MC[pID].astype('bool').astype('int')
    j=1
    for objectnumbers in [top_c, top_e, tail_c, tail_e]:
        select_fibrils_im= j * np.isin(MC[pID],objectnumbers+1).astype('int')
        im+=select_fibrils_im
        j+=1
    return im
def color_image(pID):
        im=create_image(pID)
        im_color=np.zeros([im.shape[0], im.shape[0], 3])
        def make_colormap():
            def rgb(r, g, b):
                return np.array([r, g, b])/255

            bgcolor=rgb(0, 0, 0)
            fibcol=rgb(116, 116, 116)
            top_centre=rgb(34, 63, 213)
            tail_center=rgb(255, 130, 0)

            top_edge=rgb(111, 216, 255)
            tail_edge=rgb(250, 230, 46)

            top_edge=fibcol ; tail_edge=fibcol

            return [bgcolor, fibcol,top_centre, top_edge, tail_center, tail_edge ]

        colors=make_colormap()
        for i in range(d.npix):
            for j in range(d.npix):
                im_color[i, j]= colors[im[i,j]]
        return im_color
def endsanimation():
    fig, ax1=plt.subplots()

    container = [];
    for pID in tqdm(range(d.nP)):
        im=ax1.imshow(color_image(pID), animated=True)
        plot_title = ax1.text(0.5,1.05,'Plane %d of %d' % (pID, d.nP),
                 size=plt.rcParams["axes.titlesize"],
                 ha="center", transform=ax.transAxes, )
        container.append([im, plot_title])
    ani = animation.ArtistAnimation(fig, container, interval=500, blit=True)

    ani.save(dirOutputs+'stats/ends.mp4')
def ends_fig():
    fig2, ax2=plt.subplots()
    ax2.plot(np.arange(1,d.nP-1), ends_df.sumc.to_numpy()[1:-1], '-', ms=3)
    ax2.set_xlabel('Plane') ; ax2.set_ylabel('Number of fibril termini')
    plt.savefig(dirOutputs+'stats/fibril_ends_new')
    #plt.show()
fibrilends, ends_df=find_fibril_ends()
max(ends_df.sumc.to_numpy()[1:-1])
ends_fig()
#%%How many in each plane

def how_many_fibrils():
    n_fibs_list=[]

    for pID in range(d.nP):
        n_fibs_list.append(    np.count_nonzero(FR[:, pID]+1))

    fig, ax=plt.subplots()
    ax.set_xlabel('Plane');ax.set_ylabel('Number of Fibrils')
    ax.plot(n_fibs_list) ; #plt.show()
how_many_fibrils()
