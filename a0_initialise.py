import glob
import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label, regionprops,regionprops_table
import customFunctions as md
from skimage.color import label2rgb

class metadata:
    def __init__(self, name):
        self.dataset = name    # instance variable unique to each instance
        if self.dataset=='nuts-and-bolts':

            self.local_input    =       '/Users/user/dropbox-sym/1-NutsBolts/em/'
            self.local_output   =       '/Users/user/dropbox-sym/1-NutsBolts/output/local_results_0_10/'
            self.remote_input   =       '../nuts-and-bolts/three-view/'
            self.remote_output  =       '../nuts-and-bolts/results_0_695/'

        else:
            self.local_input     =       f'/Users/user/dropbox-sym/2-MechanicsPaper/em/{self.dataset}/segmented/'
            self.local_output    =       f'/Users/user/dropbox-sym/2-MechanicsPaper/output/{self.dataset}/'
            self.remote_input    =       f'../{self.dataset}/segmented/'
            self.remote_output   =       f'../{self.dataset}/output/'

        def calculate_parameters():
            self.start               =       0
            self.nP_all              =      len(glob.glob(self.local_input+'/*proc*'))

            if ('Dropbox' in os.getcwd()):              #LOCAL
                self.dirInputs   =   self.local_input
                self.dirOutputs  =   self.local_output
                self.end         =   10

            else:                                       #ON CSF
                self.dirInputs   =   self.remote_input
                self.dirOutputs  =   self.remote_output
                self.end         =   self.nP_all

            self.nP              =   self.end-self.start #Number of planes
            self.pxsize          =   pd.read_csv(glob.glob(self.dirInputs+'/*metadata*csv')[0]).pixelsize[0]
            self.junk            =   pd.read_csv(glob.glob(self.dirInputs+'/*metadata*csv')[0]).junkslices.to_numpy()
            self.dz              =   pd.read_csv(glob.glob(self.dirInputs+'/*metadata*csv')[0]).dz[0]
            self.npix            =   Image.open(glob.glob(self.dirInputs+'*proc*')[0]).size[0]
        calculate_parameters()

def create_binary_stack(d,whitefibrils=True):
    """
    imports images from given 3V directory
    has the option to switch based on whether the fibrils are black on a white background or vice versa
    """
    imagePath = sorted(glob.glob(d.dirInputs+'*.tiff'))[d.start:d.end]
    imgstack = np.array( [np.array(Image.open(img).convert('L'), dtype=np.uint16) for img in imagePath])/255
    if whitefibrils==True:
        return imgstack
    else:
        return np.logical_not(imgstack).astype(int)  #May not always be necessary to invert!
    print("Stack Created")
def compress_by_skipping(skip):
    global imgstack
    if skip>1: #Resize array and renumber junk slices if skipping slices
        keep=skip*np.arange(d.nP/skip).astype(int)
        imgstack=imgstack[keep]
        d.junk=d.junk/skip
        dz*=skip
        d.dirOutputs= d.dirInputs+'skip_%d_results/'%skip
def create_morph_comp(imgstack):
    """
    a 3d Labelled array of image stack. Named for the morphological components function in mathematica.
    In each plane, every object gets a unique label, labelled from 1 upwards. The background is labelled 0.
    """
    MC=np.zeros(imgstack.shape, dtype=np.int16) #initialising array for morph comp
    for i in range(imgstack.shape[0]):
        print(f'MC plane {i}')
        MC[i]=label(imgstack[i])
    np.save(d.dirOutputs+'morphComp', MC)
    return MC
def create_properties_table(MC):
    """
    Setting up table of properties for each plane (props) props stores (pID, objectID, property).
    it is the length of the max number of objects in any plane, and populated with zeroes.
    Everything is measured in pixels
    """
    props_ofI='centroid','orientation','area','eccentricity' # these properties are the ones to be calculated using skimage.measure
    props=np.empty((MC.shape[0], np.max(MC), len(props_ofI)+2)) #the +2 is because centroid splits into 2, and also to leave space for the feret diameter, calculated by a custom script.
    for pID in range(MC.shape[0]):
        print(f'Properties table plane {pID}')
        rprops=pd.DataFrame(regionprops_table(MC[pID], properties=props_ofI)).to_numpy() #regionprops from skimage.measure
        #print (temp.shape)
        nobj=rprops.shape[0]; # nobjects in plane
        props[pID,0:nobj, 0:5]=rprops
        props[pID,0:nobj, 5]=feret_diameters_2d(MC[pID])
    np.save(d.dirOutputs+'props', props)
    return props
    #return temp.shape
def setup_MC_props(skip=1):
    md.create_Directory(d.dirOutputs)
    #Check for previous MC/Properties tables
    if (os.path.isfile(d.dirOutputs+'morphComp.npy') & os.path.isfile(d.dirOutputs+'props.npy')): #to save time
        print(f'Loading MC/Props from {d.dirOutputs}')
        MC=np.load(d.dirOutputs+"morphComp.npy")
        props=np.load(d.dirOutputs+"props.npy")
    else:
        print("Creating MC/Props from scratch")
        imgstack=create_binary_stack(d) #import images and create binary array
        if skip>1:
            compress_by_skipping(skip)
        MC=create_morph_comp(imgstack)
        props=create_properties_table(MC)

    return MC, props

d=metadata('nuts-and-bolts')
MC, props=setup_MC_props()
