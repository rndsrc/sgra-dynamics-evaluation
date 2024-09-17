######################################################################
# Author: Rohan Dahale, Date: 14 May 2024
######################################################################

import ehtim as eh
import argparse
import os
import glob

# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', type=str, help='path of input original hdf5 file')
    p.add_argument('-o', '--output', type=str, help='path of output average frame fits file')
    p.add_argument('-t', '--truth', action='store_true', help='If hdf5 is unblurred truth')
    p.add_argument('--scat', type=str, default='none', help='onsky, deblur, dsct, none')
    return p

######################################################################
# List of parsed arguments
######################################################################

args = create_parser().parse_args()

mv = eh.movie.load_hdf5(args.input)
if args.truth:
    if args.scat!='dsct':
        mv = mv.blur_circ(fwhm_x=15*eh.RADPERUAS,fwhm_t=0)
im = mv.avg_frame()
im.save_fits(args.output)
