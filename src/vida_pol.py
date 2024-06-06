######################################################################
# Author: Rohan Dahale, Date: 13 May 2024
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
from tqdm import tqdm
import itertools 
import sys
from copy import copy

######################################################################
# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data', type=str, 
                   default='hops_3601_SGRA_LO_netcal_LMTcal_10s_ALMArot_dcal.uvfits', 
                   help='string of uvfits to data')
    p.add_argument('--truthmv', type=str, default='none', help='path of truth .hdf5')
    p.add_argument('--kinemv', type=str, default='none', help='path of kine .hdf5')
    p.add_argument('--ehtmv',  type=str, default='none', help='path of ehtim .hdf5')
    p.add_argument('--dogmv',  type=str, default='none', help='path of doghit .hdf5')
    p.add_argument('--ngmv',   type=str, default='none', help='path of ngmem .hdf5')
    p.add_argument('--resmv',  type=str, default='none', help='path of resolve .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./plot.png', help='name of output file with path')
    p.add_argument('-c', '--cores', type=int, default='64',help='number of cores to use')
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
outpath = args.outpath
cores = args.cores

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

colors = {  'truth'    : 'black', 
            'kine'     : 'xkcd:azure',
            'resolve'  : 'tab:orange',
            'ehtim'    : 'forestgreen',
            'doghit'   : 'darkviolet',
            'ngmem'    : 'crimson'
        }

labels = {  'truth'    : 'Truth',
            'kine'     : 'kine',
            'resolve'  : 'resolve',
            'ehtim'    : 'ehtim',
            'doghit'   : 'DoG-HiT',
            'ngmem'    : 'ngMEM'
        }

mfcs = {
            'truth'    : 'none',
            'kine'     : 'xkcd:azure',
            'resolve'  : 'tab:orange',
            'ehtim'    : 'forestgreen',
            'doghit'   : 'darkviolet',
            'ngmem'    : 'crimson'
        }

mss = {
            'truth'    : 10,
            'kine'     : 5,
            'resolve'  : 5,
            'ehtim'    : 5,
            'doghit'   : 5,
            'ngmem'    : 5 
        }

data = args.data
scat= args.data

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

for p in paths.keys():
    mv=eh.movie.load_hdf5(paths[p])
    iml=[mv.get_image(t) for t in times]
    file_path = args.outpath
    parts = file_path.rsplit('/', 1)

    if len(parts) == 2:
        folder, filename = parts
    os.system(f'mkdir -p {folder}/temp/')
    i=0
    for im in iml:
        if p=='truth':
            im = im.blur_circ(fwhm_i=15*eh.RADPERUAS, fwhm_pol=15*eh.RADPERUAS)
        im.save_fits(f"{folder}/temp/{times[i]}.fits")
        i=i+1
    os.system(f'realpath {folder}/temp/*.fits > {folder}/temp/filelist.txt')
    outpath_csv= outpath[:-4]+f'_{p}.csv'
    os.system(f'julia -p {cores} ./src/VLBIImagingSummaryStats.jl/scripts/summarystats/main.jl {folder}/temp/filelist.txt {outpath_csv} -s {cores}')
    os.system(f'rm -r {folder}/temp/')

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15,6), sharex=True)
fig.subplots_adjust(hspace=0.1, wspace=0.5, top=0.8, bottom=0.01, left=0.01, right=0.8)

ax[0,0].set_ylabel('$|m|_{net}$')
ax[0,1].set_ylabel('$ \\langle |m| \\rangle$')
ax[0,2].set_ylabel('$v_{net}$')
ax[0,3].set_ylabel('$ \\langle |v| \\rangle $')

ax[1,0].set_ylabel('$|\\beta_{LP,2}|$')
ax[1,1].set_ylabel('$\\angle \\beta_{LP,2}$')
ax[1,1].set_ylim(0,360)
ax[1,1].set_yticks([0,90,180,270,360])
ax[1,2].set_ylabel('$|\\beta_{CP,1}|$')
ax[1,3].set_ylabel('$ \\angle \\beta_{CP,1}$')
ax[1,3].set_ylim(0,360)
ax[1,3].set_yticks([0,90,180,270,360])

ax[1,0].set_xlabel('Time (UTC)')
ax[1,1].set_xlabel('Time (UTC)')
ax[1,2].set_xlabel('Time (UTC)')
ax[1,3].set_xlabel('Time (UTC)')

s={}
for p in paths.keys():
    s[p]  = outpath[:-4]+f'_{p}.csv'
######################################################################

    df = pd.read_csv(s[p])

    times=[]
    for t in df['files']:
        times.append(float(os.path.basename(t)[:-5]))

    m_net = df['m_net']
    m_avg = df['m_avg']
    betalp_2 = df['re_betalp_2']+ 1j*df['im_betalp_2']
    mod_betalp_2=np.abs(betalp_2)
    ang_betalp_2=np.rad2deg(np.angle(betalp_2))

    v_net = df['v_net']
    v_avg = df['v_avg']
    betacp_1 = df['re_betacp_1'] + 1j*df['im_betacp_1']
    mod_betacp_1 = np.abs(betacp_1)
    ang_betacp_1 = np.rad2deg(np.angle(betacp_1))
            
    #for i in range(len(ang_betacp_1)):
        #if ang_betacp_1[i]<0:
        #    ang_betacp_1[i] =  ang_betacp_1[i] + 360
        #if ang_betacp_1[i]>360:
        #    ang_betacp_1[i] =  ang_betacp_1[i] - 360
    ang_betacp_1 = ang_betacp_1%360
        
                
    #for i in range(len(ang_betalp_2)):
    #    if ang_betalp_2[i]<0:
    #        ang_betalp_2[i] =  ang_betalp_2[i] + 360
    #    if ang_betalp_2[i]>360:
    #        ang_betalp_2[i] =  ang_betalp_2[i] - 360
    ang_betalp_2 = ang_betalp_2%360

    mc=colors[p]
    alpha = 0.5
    lc=colors[p]
    ms=mss[p]
    mfc=mfcs[p]
    if np.sum(m_net)!=0:
        ax[0,0].plot(times, m_net,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha, label=labels[p])
        ax[0,1].plot(times, m_avg,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha)
    if np.sum(v_net)!=0:
        ax[0,2].plot(times, v_net,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha)
        ax[0,3].plot(times, v_avg,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha)
    if np.sum(mod_betalp_2)!=0:
        ax[1,0].plot(times, mod_betalp_2,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha)
        ax[1,1].plot(times, ang_betalp_2,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha)
    if np.sum(mod_betacp_1)!=0:
        ax[1,2].plot(times, mod_betacp_1,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha)
        ax[1,3].plot(times, ang_betacp_1,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1,  color=lc, alpha=alpha)

       
ax[0,0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(5.25, 1.35), markerscale=2)
plt.savefig(outpath, bbox_inches='tight', dpi=300)