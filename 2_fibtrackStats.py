import numpy as np
import matplotlib.pyplot as plt
import customFunctions as md
from random import randint
plt.rcParams['figure.figsize'] = [10, 7.5];#default plot size
plt.rcParams['font.size']=16;
plt.rcParams['lines.linewidth'] = 2.0
#----------------------------------------------------------------------------
#....................IMPORT DATA FROM FIBRIL MAPPING....................
#-------------------------------------------------------------------------------
whichdata=0;
desired_length=0.9;

resultsDir=md.find_3V_data(whichdata)+'results\\';

try:
    fib_rec_0=np.load(resultsDir+r'\fib_rec.npy') #original, import fibril record
    morphComp=np.load(resultsDir+r'\morphComp.npy')
    props=np.load(resultsDir+r'\props.npy')
except:
    print("Error, no fibrec, morphComp, props found")

nfibs_0,nplanes=fib_rec_0.shape;
npix=morphComp.shape[2];
junk=np.nonzero(np.all(fib_rec_0==-1, axis=0))[0];

#----------------------------------------------------------------------------
#....................SELECTING FIBRILS OF A CERTAIN LENGTH....................
#-------------------------------------------------------------------------------

#Q: How long are all the fibrils in the original fibril rec?
nexist_0=np.zeros(nfibs_0, dtype='int');
for i in range(nfibs_0):
    nexist_0[i]=np.max(np.nonzero(fib_rec_0[i]>-1))-np.min(np.nonzero(fib_rec_0[i]>-1))+1;
longfibs=np.where(nexist_0>nplanes*desired_length)[0]; #the indices of the long fibirls
title_='Number of entries %i, Number above 90pc %i, Percentage %.1f '%(nfibs_0, longfibs.size, 100*longfibs.size/nfibs_0);
md.my_histogram(100*nexist_0/nplanes,'Number of slices present',title=title_, nbins=20)

#Erasing fibril record for short fibrils. The only way to map between the two is using longfibs. Reindexing also!
fib_rec=fib_rec_0[longfibs];
nexist=nexist_0[longfibs];
nfibs=longfibs.size;
lengths_scaled=np.zeros(nfibs); #worm like length / n planes present
direction_unit_vectors=np.zeros((nfibs, 3));

#Q: how many fibs are we capturing in cross-section?
meanperplane=np.mean(np.apply_along_axis(np.max, 1, np.reshape(morphComp, (nplanes,npix**2 ))));
print('fraction captured in cross section', nfibs/meanperplane)

#Export Animation
md.export_animation(resultsDir,"dropped_fibril_inquiry_50to90", morphComp,nfibs,fib_rec, dt=1000)


#----------------------------------------------------------------------------
#....................CALCULATING CRITICAL STRAIN....................
#-------------------------------------------------------------------------------

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

np.save(resultsDir+r'\scaledlengths', lengths_scaled);




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
np.save(resultsDir+r'\radii', radii);
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
np.save(resultsDir+r'\area.npy', area);
my_histogram(area/10**6, 'Area (um^2)')
#%%----------------Length vs cross secitonal Area

plt.plot(lengths_scaled,area/10**6, '.r')
plt.xlabel('Normalised lengths')
plt.ylabel('Cross Sectional Area (um^2)')
plt.show();
