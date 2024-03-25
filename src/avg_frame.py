import ehtim as eh
import argparse
import os
import glob

# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--inputfolder', type=str, default='./hdf5/', help='path of input folder with original hdf5 files')
    p.add_argument('-o', '--outputfolder', type=str, default='./fits/', help='path of output folder to save average frame fits files')
    p.add_argument('-t', '--truth', action='store_true', help='If hdf5 is unblurred truth')
    return p


######################################################################
# Plotting Setup
######################################################################
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']=300

mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 18
mpl.rcParams["ytick.labelsize"] = 18
mpl.rcParams["legend.fontsize"] = 18
######################################################################

from matplotlib import font_manager
font_dirs = font_manager.findSystemFonts(fontpaths='./fonts/', fontext="ttf")
#mpl.rc('text', usetex=True)

fe = font_manager.FontEntry(
    fname='./fonts/Helvetica.ttf',
    name='Helvetica')
font_manager.fontManager.ttflist.insert(0, fe) # or append is fine
mpl.rcParams['font.family'] = fe.name # = 'your custom ttf font name'

######################################################################
# List of parsed arguments
args = create_parser().parse_args()

hdf5list = glob.glob(args.inputfolder +'/*.hdf5')

for hdf5file in hdf5list:
    mv = eh.movie.load_hdf5(hdf5file)
    if args.truth:
        mv = mv.blur_circ(fwhm_x=15*eh.RADPERUAS,fwhm_t=0)
    im = mv.avg_frame()
    im.save_fits(args.outputfolder + '/' +os.path.basename(hdf5file)[:-5]+'.fits')
    
    im.display(export_pdf=args.outputfolder + '/' +os.path.basename(hdf5file)[:-5]+'.png', has_title=False, has_cbar=False, label_type='scale')