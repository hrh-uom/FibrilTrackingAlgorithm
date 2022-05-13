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
    def __init__(self, name, minirun, a, b, c, T):
        self.dataset        = name    # instance variable unique to each instance
        self.test           = minirun

        self.a               =   a
        self.b               =   b
        self.c               =   c
        self.threshfactor    =   T

        if self.dataset=='nuts-and-bolts':
            self.end            =   10
            self.local_input    =       '/Users/user/dbox/1-NutsBolts/em/'
            self.local_output   =       f'/Users/user/dbox/1-NutsBolts/output/local_results_0_{self.end}/'
            self.local_csf_out   =      f'/Users/user/dbox/1-NutsBolts/output/csf-695/'
            self.remote_input   =       '../nuts-and-bolts/'
            self.remote_output  =       '../nuts-and-bolts/output/'
        else: #MechanicsData
            self.end            =   10
            self.local_input     =       f'/Users/user/dbox/2-mechanics-model/em/{self.dataset}/'
            self.local_output    =       f'/Users/user/dbox/2-mechanics-model/output-{self.end}/{self.dataset}/a{self.a}_b{self.b}_c{self.c}_T{self.threshfactor}/'
            self.local_csf_out   =       f'/Users/user/dbox/2-mechanics-model/csf-output/{self.dataset}/a{self.a}_b{self.b}_c{self.c}_T{self.threshfactor}/'
            self.remote_input    =       f'../{self.dataset}/'
            self.remote_output   =       f'../{self.dataset}/output/a{self.a}_b{self.b}_c{self.c}_T{self.threshfactor}/'

        def calculate_parameters():
            """
            Reads /Sets geometrical metadata about image stack, including voxel dimensions and minimum desired fibril length
            """
            self.start               =       0
            if ('Dropbox' in os.getcwd()):              #LOCAL
                self.dirInputs   =   self.local_input
                self.nP_all      =   len(glob.glob(self.dirInputs+'segmented/*'))

                if self.test:                #TESTING FROM SCRATCH
                    self.dirOutputs  =   self.local_output
                else:               #LOCALLY TESTING CSF RESULTS
                    print('Using data from remote server')
                    self.dirOutputs  =   self.local_csf_out
                    self.end         =   self.nP_all

            else:                                       #ON CSF
                self.dirInputs   =   self.remote_input
                self.dirOutputs  =   self.remote_output
                print (f'The output directory is {self.dirOutputs}')
                self.nP_all      =   len(glob.glob(self.dirInputs+'segmented/*'))
                self.end         =   self.nP_all

            self.nP              =   self.end-self.start #Number of planes
            self.pxsize          =   pd.read_csv(glob.glob(self.dirInputs+'/*metadata*csv')[0]).pixelsize[0]
            self.junk            =   pd.read_csv(glob.glob(self.dirInputs+'/*metadata*csv')[0]).junkslices.dropna().to_numpy()
            self.dz              =   pd.read_csv(glob.glob(self.dirInputs+'/*metadata*csv')[0]).dz[0]
            self.npix            =   Image.open(glob.glob(self.dirInputs+'/segmented/*')[0]).size[0]
            if self.test:
                self.frac        =   0.5
                self.l_min       =    self.frac*self.dz*self.nP
            else:
                self.l_min           =   10000
                self.frac            =   np.round((self.l_min/self.dz)/self.nP_all, 3)


        calculate_parameters()

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
    imagePath = sorted(glob.glob(d.dirInputs+'segmented/*'))[d.start:d.end]
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
    np.save(d.dirOutputs+'morphComp', MC)
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
    np.save(d.dirOutputs+'props', props)
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
    if (os.path.isfile(d.dirOutputs+'morphComp.npy') & os.path.isfile(d.dirOutputs+'props.npy')): #to save time
        print(f'Loading MC/Props from {d.dirOutputs}')
        MC=np.load(d.dirOutputs+"morphComp.npy")
        props=np.load(d.dirOutputs+"props.npy")
    else:
        print("No MC/Props found. Creating from scratch")
        imgstack=create_binary_stack(d) #import images and create binary array
        MC=create_morph_comp(imgstack,d)
        props=create_properties_table(MC, d)
    return MC, props

# if ('Dropbox' in os.getcwd()):
#     # Local run
#     minirun=True
#     dataset='9am-1R'

# Remote run -- read off jobscript file

def initialise_dataset():
    dataset = sys.argv[1]
    minirun= sys.argv[2]
    a, b, c, T= tuple([float(sys.argv[i]) for i in [3, 4, 5,6]])

    d=metadata(dataset,minirun, a, b, c, T)
    print(f'a0: Initialising FTA for Dataset {dataset}')
    MC, props=setup_MC_props(d)
    return d, MC, props
