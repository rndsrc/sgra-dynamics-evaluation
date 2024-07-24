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
    p.add_argument('--kinemv', type=str, default='none', help='path of kine .hdf5')
    p.add_argument('--ehtmv',  type=str, default='none', help='path of ehtim .hdf5')
    p.add_argument('--dogmv',  type=str, default='none', help='path of doghit .hdf5')
    p.add_argument('--ngmv',   type=str, default='none', help='path of ngmem .hdf5')
    p.add_argument('--resmv',  type=str, default='none',help='path of resolve .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./chi2.png', 
                   help='name of output file with path')
    p.add_argument('--pol',  type=str, default='I',help='I,Q,U,V')
    p.add_argument('--scat', type=str, default='none', help='sct, dsct, none')

    return p
######################################################################
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

######################################################################

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,6), sharex=True)

ax[0].set_ylabel('$\chi^{2}$ cphase')
ax[1].set_ylabel('$\chi^{2}$ logcamp')
ax[2].set_ylabel('$\chi^{2}$ amp')
ax[0].set_xlabel('Time (UTC)')
ax[1].set_xlabel('Time (UTC)')
ax[2].set_xlabel('Time (UTC)')

ax[0].set_ylim(0.1,1e6)
ax[1].set_ylim(0.1,1e7)
ax[2].set_ylim(0.1,1e8)

mv_chi={}

j=1
for p in polpaths.keys():
    mv=eh.movie.load_hdf5(polpaths[p])
    
    imlist = [mv.get_image(t) for t in times]
    
    new_movie = eh.movie.merge_im_list(imlist)
    new_movie.reset_interp(bounds_error=False)
    
    mv_chi[p]=[]
    mv_chicp=obs.chisq(new_movie, dtype='cphase', pol=pol, ttype='fast')
    
    mv_chicp=np.round(mv_chicp,2)
    mv_chi[p].append(mv_chicp)
    mv_chilca=obs.chisq(new_movie, dtype='logcamp', pol=pol, ttype='fast')
    mv_chilca=np.round(mv_chilca,2)
    mv_chi[p].append(mv_chilca)
    mv_chia=obs.chisq(new_movie, dtype='amp', pol=pol, ttype='fast')
    mv_chia=np.round(mv_chia,2)
    mv_chi[p].append(mv_chia)

    chicp_t=[]
    chilca_t=[]
    chia_t=[]
    
    i=0
    for im in imlist:
        chicp=obslist_t[i].chisq(im, dtype='cphase', pol=pol, ttype='fast')
        chilca=obslist_t[i].chisq(im, dtype='logcamp', pol=pol, ttype='fast')
        chia=obslist_t[i].chisq(im, dtype='amp', pol=pol, ttype='fast')
            
        chicp_t.append(chicp*j)
        chilca_t.append(chilca*j)
        chia_t.append(chia*j)
        i=i+1
        
    j=j*10
                
    mc=colors[p]
    alpha = 0.5
    lc=colors[p]
    ax[0].plot(times, chicp_t,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1,  color=lc, alpha=alpha, label=labels[p])
    ax[0].set_yscale('log')
    ax[1].plot(times, chilca_t, marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1,  color=lc, alpha=alpha)
    ax[1].set_yscale('log')
    ax[2].plot(times, chia_t, marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1,  color=lc, alpha=alpha)
    ax[2].set_yscale('log')
  
k=0      
for p in polpaths.keys():
    ax[0].hlines(10**k, xmin=10.5, xmax=14.5, color=colors[p], ls='--', lw=1.5, zorder=0)
    ax[1].hlines(10**k, xmin=10.5, xmax=14.5, color=colors[p], ls='--', lw=1.5, zorder=0)
    ax[2].hlines(10**k, xmin=10.5, xmax=14.5, color=colors[p], ls='--', lw=1.5, zorder=0)
    k=k+1

    ax[0].yaxis.set_ticklabels([])
    ax[1].yaxis.set_ticklabels([])
    ax[2].yaxis.set_ticklabels([])

ax[0].legend(ncols=len(polpaths.keys()), loc='best',  bbox_to_anchor=(3., 1.2), markerscale=5.0)
ax[0].text(10.5, 5e6, f'Stokes: {pol}', color='black', fontsize=18)

col_labels =  polpaths.keys()
row_labels = ['$\chi^{2}$ cphase','$\chi^{2}$ logcamp','$\chi^{2}$ amp']
table_vals = pd.DataFrame(data=mv_chi, index=row_labels)
table = ax[1].table(cellText=table_vals.values,
                     rowLabels=table_vals.index,
                     colLabels=table_vals.columns,
                     cellLoc='center',
                     loc='bottom',
                     bbox=[-0.66, -0.5, 2.5, 0.3])

table.auto_set_font_size(False)
table.set_fontsize(18)
for c in table.get_children():
    c.set_edgecolor('none')
    c.set_text_props(color='black')
    c.set_facecolor('none')
    c.set_edgecolor('black')
        
plt.savefig(args.outpath+'.png', bbox_inches='tight', dpi=300)