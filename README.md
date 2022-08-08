There are 3 main files in the FTA (a0 a1 a2) and 3 others (b1-3) which create various figures using the data created in a0-2.

a0 is an initialisation script, which runs at the start of all the others. It reads the files (SBF-SEM .tif stack and accompanying metadata file). The metadata file is a list of broken planes (noted manually by the user), and the dimensions of each voxel, which must be stored in the same directory as the SBF-SEM planes. a0 creates a class d to store the pixel size, slice thickness, number of planes and all  the various directories for reading and writing to/from. It also either reads or creates the properties table (using scikit-image to measure the objects in all of the planes) and morphcomp, which is a stack of planes where each object (in-plane) is given a label.

a1 is the tracking algorithm, which saves the unfiltered fibril record as fib_rec.npy.

a2 reads fib_rec.npy, removes the short fibrils then calculates various statistics on the fibril population, including fibril length, orientation, helicity, etc.

b1 creates two schematics which explain how the algorithm works (1(d) 4(a))

b2 creates volume render and animation of the tracked fibrils

b3 created the heatmap and associated volumes in figure 4: Parameter optimisation

feret_diameter is a file which calculates the MFD of an object using a custom script

customFunctions is a file containing various functions used across the other scripts, imported as 'md'.

A test dataset (5 planes) with a metadata file can be found in fta-repo/testdata. The directories in a0 point to this.

Some abbreviations/terms used in the comments of the code:

SEGMENT/OBJECT-an object in a single 3View plane. May be a fibril or some other matrix material, or cellular matter
STRAND - some consecutive jointed segments, mapped by the algorithm
FIBRIL - strands which are deemed to be fibrils. This is generally done by demonstrating they stretch the full length of the volume
MFD - Minimum feret diameter. Defined as the smallest diameter of the convex hull
