######################################################################
# Author: Ilje Cho, Rohan Dahale, Date: 12 July 2024
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
from utilities import *
colors, titles, labels, mfcs, mss = common()

# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data', type=str, 
                   default='hops_3601_SGRA_LO_netcal_LMTcal_10s_ALMArot_dcal.uvfits', 
                   help='string of uvfits to data to compute chi2')
    p.add_argument('--kinemv', type=str, default='none', help='path of kine .hdf5')
    p.add_argument('--ehtmv',  type=str, default='none', help='path of ehtim .hdf5')
    p.add_argument('--dogmv',  type=str, default='none', help='path of doghit .hdf5')
    p.add_argument('--ngmv',   type=str, default='none', help='path of ngmem .hdf5')
    p.add_argument('--resmv',  type=str, default='none',help='path of resolve .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./chi2.png', 
                   help='name of output file with path')
    p.add_argument('--pol',  type=str, default='I',help='I,Q,U,V')
    p.add_argument('--scat', type=str, default='none', help='onsky, deblur, dsct, none')

    return p

# List of parsed arguments
args = create_parser().parse_args()

outpath = args.outpath
pol = args.pol

paths={}

if args.kinemv!='none':
    paths['kine']=args.kinemv
if args.resmv!='none':
    paths['resolve']=args.resmv
if args.ehtmv!='none':
    paths['ehtim']=args.ehtmv
if args.dogmv!='none':
    paths['doghit']=args.dogmv 
if args.ngmv!='none':
    paths['ngmem']=args.ngmv
    
######################################################################

obs = eh.obsdata.load_uvfits(args.data)
obs, times, obslist_t, polpaths = process_obs(obs, args, paths)

if args.pol=='I':
    clphs = pd.DataFrame(obs.c_phases(count='max', vtype='vis'))
elif args.pol=='Q':
    clphs = pd.DataFrame(obs.c_phases(count='max', vtype='qvis'))
elif args.pol=='U':
    clphs = pd.DataFrame(obs.c_phases(count='max', vtype='uvis'))
elif args.pol=='V':
    clphs = pd.DataFrame(obs.c_phases(count='max', vtype='vvis'))
else:
    print('Parse vaild pol string value: I, Q, U, V')

######################################################################

tri_list = [('AZ', 'LM', 'SM'), ('AA', 'AZ', 'SM'), ('AA', 'LM', 'SM')]

######################################################################
mov_clphs = dict()

for p in polpaths.keys():
    mv = eh.movie.load_hdf5(polpaths[p])
    
    clphs_mod_time = dict()
    clphs_mod_win = dict()

    for tri in tri_list:
        clphs_time, clphs_window = [], []
        for ii in range(len(times)):
            tstamp = times[ii]
            im = mv.get_image(times[ii])
            im.rf = obslist_t[ii].rf
            if im.xdim%2 == 1:
                im = im.regrid_image(targetfov=im.fovx(), npix=im.xdim-1)
                im.rf=obslist_t[ii].rf
                im.ra=obslist_t[ii].ra
                im.dec=obslist_t[ii].dec
            obs_mod = im.observe_same(obslist_t[ii], add_th_noise=False, ttype='fast')

            # closure phase
            if args.pol=='I':
                clphs_mod = pd.DataFrame(obs_mod.c_phases(count='max', vtype='vis'))
            elif args.pol=='Q':
                clphs_mod = pd.DataFrame(obs_mod.c_phases(count='max', vtype='qvis'))
            elif args.pol=='U':
                clphs_mod = pd.DataFrame(obs_mod.c_phases(count='max', vtype='uvis'))
            elif args.pol=='V':
                clphs_mod = pd.DataFrame(obs_mod.c_phases(count='max', vtype='vvis'))
            else:
                print('Parse vaild pol string value: I, Q, U, V')
    

            # select triangle
            subtab  = select_triangle(clphs_mod, tri[0], tri[1], tri[2])
            try:
                idx = np.where(np.round(subtab['time'].values,3)  == np.round(tstamp,3))[0][0]                
                clphs_time.append(subtab['time'][idx])
                clphs_window.append(subtab['cphase'][idx])
            except:
                pass

        clphs_mod_time[tri] = clphs_time
        clphs_mod_win[tri] = clphs_window
        
    mov_clphs[p] = [clphs_mod_time, clphs_mod_win]
######################################################################

ctab = copy(clphs)
fix_xax = True

numplt = 3
xmin = min(ctab['time'].values)
xmax = max(ctab['time'].values)

fig = plt.figure(figsize=(21,4))
fig.subplots_adjust(wspace=0.3)

axs = []
for i in range(1,numplt+1):
    axs.append(fig.add_subplot(1,3,i))

for i in tqdm(range(numplt)):
    # closure phase
    subtab  = select_triangle(ctab, tri_list[i][0], tri_list[i][1], tri_list[i][2])
    axs[i].errorbar(subtab['time'], subtab['cphase'], yerr=subtab['sigmacp'],
                    c='black', mec='black', marker='o', ls="None", ms=5, alpha=0.5)
    
    # Model
    for pipe in polpaths.keys():
        mv = eh.movie.load_hdf5(polpaths[p])
        clphs_mod_time, clphs_mod_win = mov_clphs[pipe]

        # plot
        axs[i].errorbar(clphs_mod_time[tri_list[i]], clphs_mod_win[tri_list[i]], 
                        c=colors[pipe], marker='o', ms=2.5, ls="none", label=labels[pipe], alpha=0.5)
    

    axs[i].set_title("%s-%s-%s" %(tri_list[i][0], tri_list[i][1], tri_list[i][2]), fontsize=18)
    #if fix_yax:
    axs[i].set_ylim(-200, 200)
    #axs[i].grid()
    if i == 0:
        axs[i].legend(ncols=6, loc='best',  bbox_to_anchor=(3, 1.3), markerscale=5.0)
    axs[i].set_xlabel('Time (UTC)')
    axs[i].set_ylabel('cphase')

    if fix_xax:
        axs[i].set_xlim(xmin-0.5, xmax+0.5)

axs[0].text(10.5, 260, f'Stokes: {pol}', color='black', fontsize=18)

fig.subplots_adjust(top=0.93)
plt.savefig(args.outpath+'.png', bbox_inches='tight', dpi=300)