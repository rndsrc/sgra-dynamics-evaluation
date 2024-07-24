######################################################################
# Author: Ilje Cho, Rohan Dahale, Date: 12 July 2024
######################################################################

#Import libraries
import numpy as np
import pandas as pd
import ehtim as eh
import ehtim.scattering.stochastic_optics as so
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
    
from preimcal import *
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
    p.add_argument('--resmv',  type=str, default='none', help='path of resolve .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./amp.png', 
                   help='name of output file with path')
    p.add_argument('--pol',  type=str, default='I',help='I,Q,U,V')
    p.add_argument('--scat', type=str, default='none', help='sct, dsct, none')

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

obs = eh.obsdata.load_uvfits(args.data)
obs, times, obslist_t, polpaths = process_obs(obs, args, paths)

amp = pd.DataFrame(obs.data)
######################################################################
    
bs_list = [('AZ', 'LM'), ('AA', 'AZ'), ('LM', 'SM')]

######################################################################
mov_amp = dict()


for p in polpaths.keys():
    mv = eh.movie.load_hdf5(polpaths[p])
    
    amp_mod_time = dict()
    amp_mod_win = dict()

    for bs in bs_list:
        amp_time, amp_window = [], []
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
            amp_mod = pd.DataFrame(obs_mod.data)
            
            # select baseline
            subtab  = select_baseline(amp_mod, bs[0], bs[1])
            try:
                idx = np.where(np.round(subtab['time'].values,3)  == np.round(tstamp,3))[0][0]                
                amp_time.append(subtab['time'][idx])                
                if args.pol=='I': amp_window.append(abs(subtab['vis'][idx]))
                if args.pol=='Q': amp_window.append(abs(subtab['qvis'][idx]))
                if args.pol=='U': amp_window.append(abs(subtab['uvis'][idx]))
                if args.pol=='V': amp_window.append(abs(subtab['vvis'][idx]))
            except:
                pass

        amp_mod_time[bs] = amp_time
        amp_mod_win[bs] = amp_window
        
    mov_amp[p] = [amp_mod_time, amp_mod_win]
######################################################################

vtab = copy(amp)
fix_xax = True

numplt = 3
xmin = min(vtab['time'].values)
xmax = max(vtab['time'].values)

if args.pol=='I': vis='vis'
if args.pol=='Q': vis='qvis'
if args.pol=='U': vis='uvis'
if args.pol=='V': vis='vvis'

if args.pol=='I': sigma='sigma'
if args.pol=='Q': sigma='qsigma'
if args.pol=='U': sigma='usigma'
if args.pol=='V': sigma='vsigma'
                
fig = plt.figure(figsize=(21,4))
fig.subplots_adjust(wspace=0.3)

axs = []
for i in range(1,numplt+1):
    axs.append(fig.add_subplot(1,3,i))

for i in tqdm(range(numplt)):
    subtab  = select_baseline(vtab, bs_list[i][0], bs_list[i][1])
    axs[i].errorbar(subtab['time'], abs(subtab[vis]), yerr=subtab[sigma],
                    c='black', mec='black', marker='o', ls="None", ms=5, alpha=0.5)

    for pipe in polpaths.keys():
        mv = eh.movie.load_hdf5(polpaths[pipe])
        amp_mod_time, amp_mod_win = mov_amp[pipe]
        
        axs[i].errorbar(amp_mod_time[bs_list[i]], amp_mod_win[bs_list[i]], 
                        c=colors[pipe], marker='o', ms=2.5, ls="none", label=labels[pipe], alpha=0.5)
    

    axs[i].set_title("%s-%s" %(bs_list[i][0], bs_list[i][1]), fontsize=18)

    if i == 0:
        axs[i].legend(ncols=6, loc='best',  bbox_to_anchor=(3, 1.3), markerscale=5.0)
    axs[i].set_xlabel('Time (UTC)')
    axs[i].set_ylabel(f'Amp: Stokes {pol}')

    if fix_xax:
        axs[i].set_xlim(xmin-0.5, xmax+0.5)

fig.subplots_adjust(top=0.93)
plt.savefig(args.outpath+'.png', bbox_inches='tight', dpi=300)