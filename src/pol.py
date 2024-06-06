######################################################################
# Author: Rohan Dahale, Date: 14 May 2024
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
    p.add_argument('--scat', type=str, default='none', help='sct, dsct, none')


    return p

# List of parsed arguments
args = create_parser().parse_args()
######################################################################
# Plotting Setup
######################################################################
#plt.rc('text', usetex=True)
import matplotlib as mpl
#mpl.rc('font', **{'family':'serif', 'serif':['Computer Modern Roman'], 'monospace': ['Computer Modern Typewriter']})
mpl.rcParams['figure.dpi']=300
#mpl.rcParams["mathtext.default"] = 'regular'
plt.rcParams["xtick.direction"]="in"
plt.rcParams["ytick.direction"]="in"
#plt.style.use('dark_background')
mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams["xtick.labelsize"] = 18
mpl.rcParams["ytick.labelsize"] = 18
mpl.rcParams["legend.fontsize"] = 18

from matplotlib import font_manager
font_dirs = font_manager.findSystemFonts(fontpaths='./fonts/', fontext="ttf")
#mpl.rc('text', usetex=True)

fe = font_manager.FontEntry(
    fname='./fonts/Helvetica.ttf',
    name='Helvetica')
font_manager.fontManager.ttflist.insert(0, fe) # or append is fine
mpl.rcParams['font.family'] = fe.name # = 'your custom ttf font name'
######################################################################

# Time average data to 60s
obs = eh.obsdata.load_uvfits(args.data)
obs.add_scans()
# From GYZ: If data used by pipelines is descattered (refractive + diffractive),
# Add 2% error and deblur original data.
if args.scat=='dsct':
    # Refractive Scattering
    #obs = obs.add_fractional_noise(0.02)
    obs = add_noisefloor_obs(obs, optype="quarter1", scale=1.0)
    # Diffractive Scattering
    sm = so.ScatteringModel()
    obs = sm.Deblur_obs(obs)

obs = obs.avg_coherent(60.0)
obs = obs.add_fractional_noise(0.01)

obs.add_scans()
times = []
for t in obs.scans:
    times.append(t[0])
obslist = obs.split_obs()
######################################################################
    
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

# Truncating the times and obslist based on submitted movies
obslist_tn=[]
min_arr=[] 
max_arr=[]
for p in paths.keys():
    mv=eh.movie.load_hdf5(paths[p])
    min_arr.append(min(mv.times))
    max_arr.append(max(mv.times))
x=np.argwhere(times>max(min_arr))
ntimes=[]
for t in x:
    ntimes.append(times[t[0]])
    obslist_tn.append(obslist[t[0]])
times=[]
obslist_t=[]
y=np.argwhere(min(max_arr)>ntimes)
for t in y:
    times.append(ntimes[t[0]])
    obslist_t.append(obslist_tn[t[0]])
######################################################################

colors = {  'truth'    : 'black',
            'kine'     : 'xkcd:azure',
            'ehtim'    : 'forestgreen',
            'doghit'   : 'darkviolet',
            'ngmem'    : 'crimson',
            'resolve'  : 'tab:orange'
        }

labels = {  'truth'    : 'Truth',
            'kine'     : 'kine',
            'ehtim'    : 'ehtim',
            'doghit'   : 'DoG-HiT',
            'ngmem'    : 'ngMEM',
            'resolve'  : 'resolve'
        }

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
