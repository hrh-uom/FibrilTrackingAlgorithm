import glob
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from skimage.measure import label, regionprops,regionprops_table
from feret_diameter import feret_diameters_2d

class metadata:
    """
    everything about the run and the dataset, to be used in a1-3

    Attributes:
    ----------
    dataset : string
        which dataset to examine
    test : bool
        whether or not we are testing on a small volume or executing for a full volume
    start : int
        which plane to start mapping from
    end : int
        which plane to finish on
    nP_all : int
        how many planes there are in this dataset (number of images)
    nP : int
        how many planes are being used in this run = end-start
    pxsize : float
        the pixel size in the image, read off the metadata file included with the EM images
    dz : float
        the spacing of planes in z, read off the metadata file included with the EM images
    junk : np.array
        planes which are to be discarded, read off the metadata file included with the EM images
    frac : float
        fraction of planes to be included to meet desired l_min
    l_min : float
        minimum length of tracked fibril to be included. This is not an arclength but a distance in z.
    a : float
        alpha - weighting for centroid error
    b : float
        beta - weighting for area error
    c : float
        gamma - weighting for MFD error
    threshfactor : float
        vary this to increase the error threshold. Default =1
    local_input : string (path)
        where are the EM files stored locally for this dataset
    local_output: string (path)
        where to write results to locally
    local_csf_out: string (path)
        where results and outputs from a remote run are stored locally
    remote_input: string (path)
        where are the EM files stored remotely for this dataset
    remote_output: string (path)
        where to write results to remotely


    dirOutputs: string (path)
        set as one of local_output, local_csf_out, remote_output, chosen by if statements
    dirInputs: string (path)
        set as one of local_input local_output depending on setup


    """
    def __init__(self, name, minirun, a, b, c, T, predictive):
        self.dataset        = name    # instance variable unique to each instance
        self.test           = minirun

        self.a               =   a
        self.b               =   b
        self.c               =   c
        self.threshfactor    =   T
        self.predictive      =  predictive

        if self.dataset == 'example':
            """
            Change these depending on your data
            """
            self.dirOutputs  =  './test_data/results/' 
            self.dirInputs  =   './test_data'
            self.metadatafilename   = self.dirInputs + '/example_metadata.csv'
            self.imagePaths = self.dirInputs + '/segmented'
            self.end            =      8
            self.start         =       0

        self.nP_all             =   len(glob.glob(self.dirInputs+'segmented/*'))
        self.nP                 =   self.end-self.start #Number of planes
        self.pxsize             =   pd.read_csv(self.metadatafilename ).pixelsize[0]
        self.junk               =   pd.read_csv(self.metadatafilename ).junkslices.dropna().to_numpy()
        self.dz                 =   pd.read_csv(self.metadatafilename ).dz[0]
        self.npix               =   Image.open(glob.glob(self.imagePaths+'/*')[0]).size[0]

        # params for Removing short fibrils as a fraction of the number of planes, then as an absolute length
        self.frac        =    0.5
        self.l_min       =    self.frac*self.dz*self.nP

def create_binary_stack(d,whitefibrils=True):
    """
    imports images from given 3V directory into a np.array

    Parameters:
    ----------
    d : class
        metadata class
    whitefibrils : bool
        switch based on whether the fibrils are black on a white background or vice versa

    Returns
    -------
    imgstack : array (int) of size (dz,npix,npix)

    """
    
    imagePath = sorted(glob.glob(d.imagePaths+'/*'))[d.start:d.end]
    npix=np.asarray(Image.open(imagePath[0])).shape[0]
    imgstack=np.zeros((len(imagePath), npix, npix), dtype=np.uint8)
    print("Making image stack")
    for i in tqdm(range(len(imagePath))):
        # print(imagePath[i])
        plane=np.asarray(Image.open(imagePath[i]))
        # print (type(plane))
        # print (f'size = {plane.nbytes}')
        imgstack[i]=plane//255
        # print(f'size = {plane.nbytes}')
    if whitefibrils==True:
        return imgstack
    else:
        return np.logical_not(imgstack).astype(int)  #May not always be necessary to invert!
def create_morph_comp(imgstack,d):
    """
    Creates a 3d Labelled array of image stack. Named for the morphological components function in mathematica.
    In each plane, every object gets a unique label, labelled from 1 upwards. The background is labelled 0.
    Saves this in the default output directory.

    Parameters:
    ----------
    imgstack : np.array (type=int)
        3D binary array of fibrils =1 and background =0

    Returns
    -------
    MC  : np.array(type=int)
    """
    MC=np.zeros(imgstack.shape, dtype=np.int16) #initialising array for morph comp
    print("Creating MC array from scratch")
    for i in tqdm(range(imgstack.shape[0])):
        MC[i]=label(imgstack[i])
    parentdir=os.path.dirname(d.dirOutputs[:-1])+'/'
    np.save(parentdir+'morphComp', MC)
    return MC
def create_properties_table(MC, d):
    """
    Setting up table of properties for each plane called
    it is the length of the max number of objects in any plane, and initially populated with zeroes.
    Everything is measured in pixels

    Parameters:
    ----------
    MC : np.array (type=int)
        a 3d Labelled array of image stack.
    Returns
    -------
    props : np array(planenumber, objectnumber, property).
        plane number from 0 to nP-1
        object number from 1 to nObj
        where properies are indexed 0-5
        'centroid' [0, 1],'orientation' [2],'area' [3],'eccentricity' [4], 'MFD' [5]
    """
    props_ofI='centroid','orientation','area','eccentricity' # these properties are the ones to be calculated using skimage.measure
    print(f'MC shape {MC.shape}')
    props=np.zeros((MC.shape[0], np.max(MC), len(props_ofI)+2)) #the +2 is because centroid splits into 2, and also to leave space for the feret diameter, calculated by a custom script.
    print(f'Making properties table plane')
    for pID in tqdm(range(MC.shape[0])):
        rprops=pd.DataFrame(regionprops_table(MC[pID], properties=props_ofI)).to_numpy() #regionprops from skimage.measure
        #print (temp.shape)
        nobj=rprops.shape[0]; # nobjects in plane
        props[pID,0:nobj, 0:5]=rprops
        props[pID,0:nobj, 5]=feret_diameters_2d(MC[pID])
    parentdir=os.path.dirname(d.dirOutputs[:-1])+'/'
    np.save(parentdir+'props', props)
    return props
    #return temp.shape
def create_Directory(directory):
    """
    Tests for existence of directory then creates one if not

    Parameters:
    ----------
    directory : string (path)
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
def setup_MC_props(d):
    """
    Checks for the existence of a morphological components array and
    a properties array, from a previous run, and makes one if it
    does not already exist

    Returns:
    ----------
    MC : np.array(int)
        as in create_morph_comp
    props : np.array (float)
        as in create_properties_table
    """
    create_Directory(d.dirOutputs)
    #Check for previous MC/Properties tables
    parentdir=os.path.dirname(d.dirOutputs[:-1])+'/'

    if (os.path.isfile(parentdir+'morphComp.npy') & os.path.isfile(parentdir+'props.npy')): #to save time
        print(f'Loading MC/Props from {parentdir}')
        MC=np.load(parentdir+'morphComp.npy')
        props=np.load(parentdir+'props.npy')
    else:
        print(f"No MC/Props found. Creating from scratch in {d.dirOutputs}")
        imgstack=create_binary_stack(d) #import images and create binary array
        MC=create_morph_comp(imgstack,d)
        props=create_properties_table(MC, d)
    return MC, props

def initialise_dataset():
    #==========Read stuff from jobscript ============
    dataset = sys.argv[1]
    minirun= bool(int(sys.argv[2]))
    a= float(sys.argv[3])
    b, c, T= tuple([float(sys.argv[i]) for i in [4, 5,6]])
    predictive= bool(int(sys.argv[7]))
    #=========================================

    d=metadata(dataset,minirun, a, b, c, T, predictive)
    print(f'a0: Initialising FTA for Dataset {dataset}')
    MC, props=setup_MC_props(d)
    return d, MC, props

if __name__ == "__main__":
    d, _,_ =initialise_dataset()
    print(d)
