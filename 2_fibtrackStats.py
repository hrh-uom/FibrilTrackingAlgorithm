import numpy as np
import matplotlib.pyplot as plt
import fibtrack_custom_functions as md
from random import randint
plt.rcParams['figure.figsize'] = [10, 7.5];#default plot size
plt.rcParams['font.size']=16;
plt.rcParams['lines.linewidth'] = 2.0
#----------------------------------------------------------------------------
#....................IMPORT DATA FROM FIBRIL MAPPING....................
#-------------------------------------------------------------------------------
whichdata=9.100;
desired_length=0.9;

if whichdata==9:
    mainDir = r'D:\3View\9am-achilles-fshx\7.5um_crop_2';
    exportDir = mainDir+r'\skip_2_results'
elif whichdata==19:
    mainDir= r'D:\3View\7pmAchx700\7-5um'
    exportDir = mainDir+'\skip_3_results'
elif whichdata==19.100:
    mainDir= r'D:\3View\7pmAchx700\7-5um\dummy_0_100'
    exportDir = mainDir+r'\results'
elif whichdata==9.15: #dummyset 15 planes!
    mainDir=r'D:\3View\9am-achilles-fshx\7.5um_crop_2\dummy_110_124'
    exportDir=mainDir+r'\results'
elif whichdata==9.100: #dummyset 100
    mainDir=r'D:\3View\9am-achilles-fshx\7.5um_crop_2\dummy_100_200'
    exportDir=mainDir+r'\results'
else:
    x=1; #dummy
try:
    fib_rec_0=np.load(exportDir+r'\fib_rec.npy'); #original, import fibril record
except:
    print("No fibril record found")
try:
    pxsize, dz=np.genfromtxt(mainDir+'\pxsize.csv', delimiter=',')[1];
except:
    print("No pixelsize file")
try:
    morphComp=np.load(exportDir+r'\morphComp.npy');
except:
    print("No fibril record found")
try:
    props=np.load(exportDir+r'\props.npy');
except:
    print("No properties found")

nfibs_0,nplanes=fib_rec_0.shape;
npix=morphComp.shape[2];
junk=np.nonzero(np.all(fib_rec_0==-1, axis=0))[0];

def my_histogram(arr,xlabel, title='Histogram',  nbins=10):
    """
    For use in future plots, a histogram!
    """
    n, bins, patches = plt.hist(arr, nbins, density=False, facecolor='g', alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel('Number')
    plt.title(title)
    plt.grid(True)
    plt.show()
#----------------------------------------------------------------------------
#....................SELECTING FIBRILS OF A CERTAIN LENGTH....................
#-------------------------------------------------------------------------------

#Q: How long are all the fibrils in the original fibril rec?
nexist_0=np.zeros(nfibs_0, dtype='int');
for i in range(nfibs_0):
    nexist_0[i]=np.max(np.nonzero(fib_rec_0[i]>-1))-np.min(np.nonzero(fib_rec_0[i]>-1))+1;
longfibs=np.where(nexist_0>nplanes*desired_length)[0]; #the indices of the long fibirls
title_='Number of entries %i, Number above 90pc %i, Percentage %.1f '%(nfibs_0, longfibs.size, 100*longfibs.size/nfibs_0);
my_histogram(100*nexist_0/nplanes,'Number of slices present',title=title_, nbins=20)


#Erasing fibril record for short fibrils. The only way to map between the two is using longfibs. Reindexing also!
fib_rec=fib_rec_0[longfibs];
nexist=nexist_0[longfibs];
nfibs=longfibs.size;
lengths_scaled=np.zeros(nfibs); #worm like length / n planes present
direction_unit_vectors=np.zeros((nfibs, 3));

#Q: how many fibs are we capturing in cross-section?
meanperplane=np.mean(np.apply_along_axis(np.max, 1, np.reshape(morphComp, (nplanes,npix**2 ))));
print('fraction captured in cross section', nfibs/meanperplane)


#%%

#Q: which ones are nearly full length but not quite 6/11/2020
half_length_fibril_indices=np.nonzero((0.5*nplanes<nexist_0)&(nexist_0<=desired_length*nplanes))[0]
#md.view_fibs_inline_2(morphComp,half_length_fibril_indices, fib_rec_0, nplanes, startplane=0,save=False, axis=3)



##%%
#Q: At what point are we losing fibrils?
half_and_above=longfibs=np.where(nexist_0>nplanes*0.5)[0]
fib_rec_0.shape
plt.plot(np.count_nonzero(fib_rec_0[half_and_above]>-1, axis=0), '-b')
plt.xlabel("Plane (/100)");
plt.ylabel("Nfibrils");
plt.title("Segments 50% of planes or more");
plt.show()

##%%
#Q: where are the fibril ends?
fibril_ends=[];

for fID in half_and_above:
    temp=np.nonzero(fib_rec_0[fID]>-1)[0];
    fibril_ends.extend([temp[0],temp[-1]]);
fibril_ends=np.array(fibril_ends);
N=fibril_ends.size
fibril_ends=fibril_ends[fibril_ends>0]
fibril_ends=fibril_ends[fibril_ends<100]
n, bins, patches = plt.hist(fibril_ends, nplanes, density=False, facecolor='g', alpha=0.75)
plt.xlabel("Plane")
plt.ylabel('Number')
plt.ylim(0,45)
plt.vlines(junk+0.5, 0, 1000);
plt.title("Location of Fibril ends, 50% and longer segments excluding 0 and 100")
plt.grid(True, )
plt.show()
print("%i out of %i are 0 and 100, thats %lf percent" % (N-fibril_ends.size,N,100* (N-fibril_ends.size)/N))


##%%
help(md.export_mapping_animation)

exportDir=r"C:\Users\t97721hr\Dropbox (The University of Manchester)\Fibril Tracking Algorithm\dropped_fibril_inquiry"
md.export_mapping_animation(exportDir, morphComp,half_length_fibril_indices,fib_rec_0,nplanes, 0, '\\50-90PC_fibs_9100_dataset')


#%%
#All selected fibs
md.view_fibs_inline_2(morphComp,np.arange(nfibs), fib_rec, nplanes, startplane=0,save=False, axis=3)



#%%STOPLINE 6th Nov

def fascicleCoord(pID):
    """
    Calculates the mean co-ordinate of all fibrils at slice pID
    """
    objects_in_plane=fib_rec[:, pID][ fib_rec[:, pID]>-1];
    if objects_in_plane.size!=0: #ignoring junk slices
        coOrds_2D=[];
        for i in objects_in_plane:
            coOrds_2D.append((props[pID, i, 0:2]));

        return np.append(pxsize*np.mean(np.array(coOrds_2D), axis=0), dz*pID);
def fibCoords(i):
    """
    calculates the coordinates in 3d real space of a fibril (i).
    """
    co_ords=np.full((nplanes, 3),-1.); #an array of the centroid co-ordinates for each fibril
    for pID in range(nplanes):
        if fib_rec[i, pID]!=-1:
            co_ords[pID, 0:2]=(props[pID, fib_rec[i,pID], 0:2])*pxsize;
            co_ords[pID, 2]=pID*dz;
    co_ords=co_ords[co_ords[:,2]>-1.]; #getting rid of junk slices / places where absent
    #This stuff is to draw a line of best fit! Leaving it out
    #mean = np.mean(co_ords, axis=0);
    #uu, dd, vv=np.linalg.svd(co_ords-mean)
    #direction=vv[0];
    return co_ords;#, mean, direction;
def plotfibril_withLOBF(i):
    co_ords, mean, direction=calculaterr_coordinates(i);
    linepts =0.5*np.linalg.norm(co_ords[0]-co_ords[-1])* direction *np.mgrid[-1:1:2j][:, np.newaxis];
    linepts += mean;# shift by the mean to get the line in the right place
    ax = m3d.Axes3D(plt.figure())
    ax.scatter3D(*co_ords.T);
    ax.plot3D(*linepts.T);
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y',fontsize=20,)
    ax.set_zlabel('z', fontsize=20 )
    ax.view_init(elev=50, azim=160)
    plt.show()
def coOrds_to_length(co_ords):
    """
    input an Nx3 numpy array representing a list of 3D coordinates, and it will calculate the length of the 'worm like' length joining all the co-ordinates in 3d
    """
    L=0;
    for j in range((co_ords.shape[0])-1): #j steps through planes in co-ords
        dr=co_ords[j]-co_ords[j+1];
        L+=np.linalg.norm(dr);
    return L;

#Calculate fascicle length
fas_coord_list=[];
for pID in range (nplanes-1):
    if np.all(fascicleCoord(pID))!=None:
        fas_coord_list.append(fascicleCoord(pID));
fas_len=coOrds_to_length(np.array(fas_coord_list));


#Calculate length of each fibril
for i in range (nfibs):
    lengths_scaled[i]=coOrds_to_length(fibCoords(i));
lengths_scaled*=nplanes/(fas_len*nexist);
#plotfibril_withLOBF(randint(nfibs))


#How long are the long fibrils?
my_histogram(lengths_scaled, 'Length (scaled by number of slices present)', title='');

np.save(exportDir+r'\scaledlengths', lengths_scaled);




#%%---------------------------Radius of each fibril

def fibril_radius(i): #maps between props and fibrec
    radii=np.full(nplanes,-1.); #an array of the centroid co-ordinates for each fibril
    for pID in range(nplanes):
        if fib_rec[i, pID]!=-1:
            radii[pID]=(props[pID, fib_rec[i,pID], 5])*pxsize;
    radii=radii[radii>-1.]; #getting rid of junk slices / places where absent
    mean = np.mean(radii, axis=0);
    return mean;
radii=np.array([fibril_radius(i) for i in range(nfibs)])
np.save(exportDir+r'\radii', radii);
my_histogram(radii, 'Radius (nm)')


#%% ------------------------Area of each fibrils

def fibril_area(i): #maps between props and fibrec
    area=np.full(nplanes,-1.); #an array of the centroid co-ordinates for each fibril
    for pID in range(nplanes):
        if fib_rec[i, pID]!=-1:
            area[pID]=(props[pID, fib_rec[i,pID],3])*(pxsize**2);
    area=area[area>-1.]; #getting rid of junk slices / places where absent
    mean = np.mean(area, axis=0);
    return mean;
area=np.array([fibril_area(i) for i in range(nfibs)])
np.save(exportDir+r'\area.npy', area);
my_histogram(area/10**6, 'Area (um^2)')
#%%----------------Length vs cross secitonal Area

plt.plot(lengths_scaled,area/10**6, '.r')
plt.xlabel('Normalised lengths')
plt.ylabel('Cross Sectional Area (um^2)')
plt.show();
