


# Getting started

The following instructions are written for Mac/Unix/Linux users.  If you are on windows, you can install [Cygwin](https://www.cygwin.com/) or [Git Bash](https://gitforwindows.org/) to load a Unix-like environment, then follow these instructions.


## Setup

Copy this repo locally. Do this by running

`git clone https://github.com/hrh-uom/FibrilTrackingAlgorithm.git`

in the terminal, in the folder where you would like it to be cloned.

You will need python 3.7 or later with pip or conda.

the requirements are standard scientific packages, if you want to install everything at once, navigate to the folder of this repo in your terminal and type

`pip install requirements.txt` or `conda install requirements.txt`


## How to run the FTA?

 - The FTA is run by running `local.sh` in the terminal with the command
		      `bash local.sh`
	>Ensure you are into the correct project folder.  
  - There are three main scripts which do the computations in the FTA. They are numbered a0-2.
  - The three other scripts are for analysis (b1-3) which create various
   figures using the data created by a0-2.
   - Inside local.sh, you may edit required values of a, b, c (weights in
   the error function) and T (error threshold.)

 > Inside `local.sh`, you may need to change `python` to `python3` if you are using a mac.

## Preprocessing the data

 - The SBF-SEM images will need to be binarised, segmented and in .tif format before running the FTA.
 - This can be completed with any image analysis software.
	>The authors like [FIJI](https://fiji.sc/).
# The scripts and what they do
## a0: Initialisation
- `a0` is an initialisation script, which runs at the start of all the others. It reads the files (SBF-SEM .tif stack and accompanying metadata file).
- The metadata file is a list of broken planes (noted manually by the user, numbered from 0), and the dimensions of each voxel, which must be stored in the same directory as the SBF-SEM planes.
- `a0` creates a class d to store the pixel size, slice thickness, number of planes and all  the various directories for reading and writing to/from.
-  It also either reads or creates the properties table (using
`scikit-image` to measure the objects in all of the planes) and
morphcomp, which is a stack of planes where each object (in-plane) is
given a label. These values and arrays are used in other scripts `a1-2`
and `b1-3`.

   >A test segmented dataset (5 planes) with a metadata file can be found in fta-repo/testdata. The directories in a0 point to this.

- `a0` returns an object `d`, (short for 'dataset'), which contains much of
   the information about the dataset you're working with.

## a1: Tracking
`a1` is the tracking algorithm, which writes the fibril record as `fib_rec.npy`. This is the **unfiltered** fibril record (short fibrils are not yet discarded).

## a2: Statistics
 - `a2` reads `fib_rec.npy`, removes the short fibrils, and writes a new copy of the fibril record out (called `fib_rec_trim.npy`).
- `a2`then calculates various statistics on the fibril population, including fibril length, orientation, chirality, etc. The plots are written to a folder labeled 'results'.

## b1: Explanatory schematics
`b1` creates two schematics which explain how the algorithm works (1(d) 4(a))

## b2: Volume rendering
`b2` creates volume render and animation of the tracked fibrils

## b3: Error function parameter optimisation
`b3` created the heatmap and associated volumes in figure 4: Parameter optimisation

# Other files
* `feret_diameter` is a file which calculates the MFD of an object, as this is not included in other packages used.
* `customFunctions` is a file containing various functions used across the other scripts, imported as 'md'.

---
**Some abbreviations/terms used in the comments of the code:**

* Segment/object : an object in a single 3View plane. May be a fibril or some other matrix material, or cellular matter
* Strand : some consecutive jointed segments, mapped by the algorithm
* Fibrils : strands which are deemed to be fibrils. This is generally done by demonstrating they stretch the full length of the volume
* MFD : Minimum feret diameter. Defined as the smallest diameter of the convex hull
