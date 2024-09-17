######################################################################
# Author: Rohan Dahale, Date: 12 July 2024
######################################################################

# Import libraries
import numpy as np
import pandas as pd
import ehtim as eh
import ehtim.scattering.stochastic_optics as so
from preimcal import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    p.add_argument('--truthmv',  type=str, default='none', help='path of truth .hdf5')
    p.add_argument('--kinemv',  type=str, default='none', help='path of kine .hdf5')
    p.add_argument('--ehtmv',   type=str, default='none', help='path of ehtim .hdf5')
    p.add_argument('--dogmv',   type=str, default='none', help='path of doghit .hdf5')
    p.add_argument('--ngmv',    type=str, default='none', help='path of ngmem .hdf5')
    p.add_argument('--resmv',   type=str, default='none', help='path of resolve .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./chi2.png', 
                   help='name of output file with path')
    p.add_argument('--scat', type=str, default='none', help='onsky, deblur, dsct, none')


    return p

# List of parsed arguments
args = create_parser().parse_args()

pathmovt = args.truthmv
outpath = args.outpath

paths={}
if args.truthmv!='none':
    paths['truth']=args.truthmv
if args.kinemv!='none':
    paths['kine']=args.kinemv
if args.ehtmv!='none':
    paths['ehtim']=args.ehtmv
if args.dogmv!='none':
    paths['doghit']=args.dogmv 
if args.ngmv!='none':
    paths['ngmem']=args.ngmv
if args.resmv!='none':
    paths['resolve']=args.resmv
    
######################################################################

obs = eh.obsdata.load_uvfits(args.data)
obs, times, obslist_t, polpaths = process_obs(obs, args, paths)

######################################################################

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(32,6), sharex=True)

ax[0].set_ylabel('$|m|_{net}$')
ax[1].set_ylabel('$ \\langle |m| \\rangle$')
ax[2].set_ylabel('$v_{net}$')
ax[3].set_ylabel('$ \\langle |v| \\rangle $')

ax[0].set_xlabel('Time (UTC)')
ax[1].set_xlabel('Time (UTC)')
ax[2].set_xlabel('Time (UTC)')
ax[3].set_xlabel('Time (UTC)')

"""
mvt=eh.movie.load_hdf5(pathmovt)
if args.scat=='dsct':
    mvt=mvt.blur_circ(fwhm_x=15*eh.RADPERUAS, fwhm_x_pol=15*eh.RADPERUAS, fwhm_t=0)
"""
        
polpaths={}
for p in paths.keys():
    mv=eh.movie.load_hdf5(paths[p])
    im=mv.im_list()[0]
    if len(im.ivec)>0 and len(im.qvec)>0 and len(im.uvec)>0 :
        polpaths[p]=paths[p]

polvpaths={}
for p in paths.keys():
    mv=eh.movie.load_hdf5(paths[p])
    im=mv.im_list()[0]
    if len(im.ivec)>0 and len(im.vvec)>0:
        polvpaths[p]=paths[p]
            
for p in polpaths.keys():
    mv=eh.movie.load_hdf5(polpaths[p])
    
    imlist = [mv.get_image(t) for t in times]
    mnet_t=[]
    mavg_t=[]
    
    for im in imlist:
        imi=im.ivec
        q=im.qvec
        u=im.uvec
        mnet = np.sqrt(np.sum(q)**2 + np.sum(u)**2)/np.sum(imi)
        mnet_t.append(mnet)
        mavg = np.sum(np.sqrt(q**2 + u**2))/np.sum(imi)
        mavg_t.append(mavg)
    
    mc=colors[p]
    alpha = 0.5
    lc=colors[p]
    
    ax[0].plot(times, mnet_t,  marker ='o', mfc=mc, mec=mc, ms=5, ls='-', lw=1,  color=lc, alpha=alpha, label=labels[p])
    ax[1].plot(times, mavg_t,  marker ='o', mfc=mc, mec=mc, ms=5, ls='-', lw=1,  color=lc, alpha=alpha)

for p in polvpaths.keys():
    mv=eh.movie.load_hdf5(polvpaths[p])
    
    imlist = [mv.get_image(t) for t in times]
    vnet_t=[]
    vavg_t=[]
    
    for im in imlist:
        imi=im.ivec
        v=im.vvec
        vnet = np.sum(v)/np.sum(imi)
        vnet_t.append(vnet)
        vavg = np.sum(abs(v/imi)*imi)/np.sum(imi)
        vavg_t.append(vavg)
    
    mc=colors[p]
    alpha = 0.5
    lc=colors[p]
    
    ax[2].plot(times, vnet_t,  marker ='o', mfc=mc, mec=mc, ms=5, ls='-', lw=1,  color=lc, alpha=alpha)
    ax[3].plot(times, vavg_t,  marker ='o', mfc=mc, mec=mc, ms=5, ls='-', lw=1,  color=lc, alpha=alpha)

ax[0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(3.5, 1.2), markerscale=2.5)
plt.savefig(args.outpath+'.png', bbox_inches='tight', dpi=300)
