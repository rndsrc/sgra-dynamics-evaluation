######################################################################
# Author: Rohan Dahale, Date: 12 July 2024
######################################################################

import ehtim as eh
import ehtim.scattering.stochastic_optics as so
from preimcal import *
import ehtplot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.cm import ScalarMappable
import pdb
import argparse
import os
import glob
from utilities import *
colors, titles, labels, mfcs, mss = common()

# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data', type=str, 
                   default='hops_3601_SGRA_LO_netcal_LMTcal_10s_ALMArot_dcal.uvfits', 
                   help='string of uvfits to data to compute chi2')
    p.add_argument('--truthmv', type=str, default='none', help='path of truth .hdf5')
    p.add_argument('--kinemv', type=str, default='none', help='path of kine .hdf5')
    p.add_argument('--starmv', type=str, default='none', help='path of starwarps .hdf5')
    p.add_argument('--ehtmv',  type=str, default='none', help='path of ehtim .hdf5')
    p.add_argument('--dogmv',  type=str, default='none', help='path of doghit .hdf5')
    p.add_argument('--ngmv',   type=str, default='none', help='path of ngmem .hdf5')
    p.add_argument('--resmv',  type=str, default='none',help='path of resolve .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./gif.gif', 
                   help='name of output file with path')
    p.add_argument('--scat', type=str, default='none', help='onsky, deblur, dsct, none')

    return p

# List of parsed arguments
args = create_parser().parse_args()

outpath = args.outpath

paths={}

if args.truthmv!='none':
    paths['truth']=args.truthmv
if args.kinemv!='none':
    paths['kine']=args.kinemv
if args.resmv!='none':
    paths['resolve']=args.resmv
if args.starmv!='none':
    paths['starwarps']=args.starmv
if args.ehtmv!='none':
    paths['ehtim']=args.ehtmv
if args.dogmv!='none':
    paths['doghit']=args.dogmv 
if args.ngmv!='none':
    paths['ngmem']=args.ngmv
    
######################################################################

obs = eh.obsdata.load_uvfits(args.data)
obs, times, obslist_t, polpaths = process_obs(obs, args, paths)
    
######################################################################
# Set parameters
npix   = 160
fov    = 160 * eh.RADPERUAS
blur   = 0 * eh.RADPERUAS
######################################################################

# Set the number of subplots
N = len(paths.keys()) 
######################################################################


df = pd.DataFrame(obs.data)
mv={}

for p in paths.keys():
    # Load the movie data and extract images at the scan times
    mvf = eh.movie.load_hdf5(paths[p])
    imlist = []
    for t in times:
        im = mvf.get_image(t).regrid_image(fov, npix)
        imlist.append(im)
    mv[p] = eh.movie.merge_im_list(imlist)

# Create subplots
fig, ax = plt.subplots(nrows=1, ncols=N, figsize=(15*N/7+1,5))
fig.subplots_adjust(hspace=0.1, wspace=0.1, top=0.7, bottom=0.1, left=0.005, right=0.9)
for i in range(N):
    ax[i].set_xticks([]), ax[i].set_yticks([])
    
var={}
imlist={}
varmax=[]
varmin=[]

for p in paths.keys():
    amplist=[]
    imlist = mv[p].im_list()
    for i in range(len(imlist)):
        im = imlist[i]

        npix = im.xdim   
        U = np.linspace(-10.0e9, 10.0e9, npix)
        V = np.linspace(-10.0e9, 10.0e9, npix)
        UU, VV = np.meshgrid(U, V)
        UV = np.vstack((UU.flatten(), VV.flatten())).T
        vis, _, _, _ = im.sample_uv(UV)
        fft = np.array(vis)

        # Calculate the amplitude of the FFT
        amp = np.abs(fft).reshape(im.xdim, im.xdim)
        amplist.append(amp)

    # Calculate the variance
    var[p] = np.std(np.array(amplist), axis=0) ** 2 * 100
    varmax.append(np.max(var[p]))
    varmin.append(np.min(var[p]))

scale = 10
i=0
for p in paths.keys():
    
    a = ax[i].imshow(var[p], cmap='binary', extent=[scale, -scale, -scale, scale], vmin=0, vmax=max(varmax))
    # Overlay the visibility points
    ax[i].plot(df['u'] / 1e9, df['v'] / 1e9, marker='o', markersize=1, color='orange', mec='orange', ls='none', zorder=2)
    ax[i].plot(-df['u'] / 1e9, -df['v'] / 1e9, marker='o', markersize=1, color='orange', mec='orange', ls='none', zorder=2)
    # Set axis limits and labels
    ax_lim = 10  # Glambda
    ax[i].set_xlim([ax_lim, -ax_lim])
    ax[i].set_ylim([-ax_lim, ax_lim])
    ax[i].set_title(titles[p])
    i=i+1
# Add colorbar to the last subplot
cax = fig.add_axes([0.01, 0.15, 0.885, 0.02])  # Adjust the position and size as needed
cb = plt.colorbar(a, cax=cax, orientation='horizontal')
cb.set_label('Visibility Variance (10$^{-2}$ Jy$^2$)')

plt.savefig(f'{outpath}.png', bbox_inches='tight')