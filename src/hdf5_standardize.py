#####################################################################
# Author: Rohan Dahale, Marianna Foschi, Date: 25 Mar 2024
######################################################################

# Import libraries
import numpy as np
import ehtim as eh
import argparse
import glob
import os

# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--inputfolder', type=str, default='./hdf5/', help='path of input folder with original hdf5 files')
    p.add_argument('-o', '--outputfolder', type=str, default='./standard_hdf5/', help='path of output folder to save standard hdf5 files')
    return p

######################################################################
# List of parsed arguments
args = create_parser().parse_args()
npix   = 128
fov    = 200 * eh.RADPERUAS
ntimes = 100
tstart = 10.90 
tstop  = 13.90 #14.04 

# tstart and tstop from obsfile of an example synthetic data
#obs = eh.obsdata.load_uvfits('model1_Ma+0.94_w3_Rh160_i30_3601_LO_tint60_syserr2_deblurTrue_reftypequarter1.uvfits')
#>>> obs.tstop
#14.041666030883789
#>>> obs.tstart
#10.891666889190674

# Create output folder
if not os.path.exists(args.outputfolder):
    os.makedirs(args.outputfolder)
    
# List of input hdf5 files
hdf5list = glob.glob(args.inputfolder +'/*.hdf5')

# Loop over all files
for hdf5file in hdf5list:
    if not os.path.exists(args.outputfolder + '/' + os.path.basename(hdf5file)):
        movie = eh.movie.load_hdf5(hdf5file)
        times = np.linspace(tstart, tstop, ntimes)

        frame_list = [movie.get_image(t).regrid_image(fov, npix) for t in times]
        new_movie = eh.movie.merge_im_list(frame_list)
        new_movie.reset_interp(bounds_error=False)
    
        new_movie.save_hdf5(args.outputfolder + '/' +os.path.basename(hdf5file))
######################################################################
