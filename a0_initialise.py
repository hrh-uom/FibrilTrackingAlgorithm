import glob
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from skimage.measure import label, regionprops,regionprops_table
from feret_diameter import feret_diameters_2d

class metadata:
    def __init__(self, name, minirun):
        self.dataset        = name    # instance variable unique to each instance
        self.test           = minirun


        if self.dataset=='nuts-and-bolts':
            self.end            =   10
            self.local_input    =       '/Users/user/dbox/1-NutsBolts/em/'
            self.local_output   =       f'/Users/user/dbox/1-NutsBolts/output/local_results_0_{self.end}/'
            self.local_csf_out   =      f'/Users/user/dbox/1-NutsBolts/output/csf-695/'
            self.remote_input   =       '../nuts-and-bolts/'
            self.remote_output  =       '../nuts-and-bolts/'
        else: #MechanicsData
            self.end            =   25
            self.local_input     =       f'/Users/user/dbox/2-MechanicsPaper/em/{self.dataset}/'
            self.local_output    =       f'/Users/user/dbox/2-MechanicsPaper/output-{self.end}/{self.dataset}/'
            self.local_csf_out   =       f'/Users/user/dbox/2-MechanicsPaper/csf-output/{self.dataset}/'
            self.remote_input    =       f'../{self.dataset}/'
            self.remote_output   =       f'../{self.dataset}/output/'

        def calculate_parameters():
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
                self.l_min       =    self.frac*self.dz*self.nP_all
            else:
                self.l_min           =   10000
                self.frac            =   np.round((self.l_min/self.dz)/self.nP_all, 3)
        calculate_parameters()

def create_binary_stack(d,whitefibrils=True):
    """
    imports images from given 3V directory
    has the option to switch based on whether the fibrils are black on a white background or vice versa
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
    print("Creating MC array from scratch")
    for i in tqdm(range(imgstack.shape[0])):
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
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def setup_MC_props(skip=1):
    create_Directory(d.dirOutputs)
    #Check for previous MC/Properties tables
    if (os.path.isfile(d.dirOutputs+'morphComp.npy') & os.path.isfile(d.dirOutputs+'props.npy')): #to save time
        print(f'Loading MC/Props from {d.dirOutputs}')
        MC=np.load(d.dirOutputs+"morphComp.npy")
        props=np.load(d.dirOutputs+"props.npy")
    else:
        print("No MC/Props found. Creating from scratch")
        imgstack=create_binary_stack(d) #import images and create binary array
        if skip>1:
            compress_by_skipping(skip)
        MC=create_morph_comp(imgstack)
        props=create_properties_table(MC)
        # props=1
    return MC, props

if ('Dropbox' in os.getcwd()):
    minirun=True
    dataset='9am-2L' # dataset='nuts-and-bolts'
else:
    dataset = sys.argv[1]
    try:
        minirun = sys.argv[2]
    except:
        minirun = False

d=metadata(dataset,minirun)
print(f'a0: Initialising FTA for Dataset {dataset}')
MC, props=setup_MC_props()
#%%

# i=np.random.randint(0, d.nP)
# # i=45;
# print(f'i = {i}')
# print(feret_diameters_2d(MC[i]))
# props[80, 22]
