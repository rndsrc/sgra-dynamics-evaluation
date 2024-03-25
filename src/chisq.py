######################################################################
# Author: Rohan Dahale, Date: 25 Mar 2024
######################################################################

# Import libraries
import numpy as np
import pandas as pd
import ehtim as eh
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
    p.add_argument('--kinemv', type=str, default='', help='path of kine .hdf5')
    p.add_argument('--starmv', type=str, default='', help='path of starwarps .hdf5')
    p.add_argument('--ehtmv',  type=str, default='', help='path of ehtim .hdf5')
    p.add_argument('--dogmv',  type=str, default='', help='path of doghit .hdf5')
    p.add_argument('--ngmv',   type=str, default='', help='path of ngmem .hdf5')
    p.add_argument('--resmv',  type=str, default='',help='path of resolve .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./chi2.png', 
                   help='name of output file with path')
    p.add_argument('--pol',  type=str, default='I',help='I,Q,U,V')
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
plt.rcParams["ytick.major.size"]=5
plt.rcParams["ytick.minor.size"]=2.5
plt.style.use('dark_background')

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
obs = obs.avg_coherent(60.0)

# From GYZ: If data used by pipelines is descattered (refractive + diffractive),
# Add 2% error and deblur original data.
if args.scat=='dsct':
    obs = obs.add_fractional_noise(0.02)
    import ehtim.scattering.stochastic_optics as so
    sm = so.ScatteringModel()
    obs = sm.Deblur_obs(obs)

obs.add_scans()
times = []
for t in obs.scans:
    times.append(t[0])
obslist = obs.split_obs()
######################################################################

pathmov  = args.kinemv
pathmov2 = args.starmv
pathmov3 = args.ehtmv
pathmov4 = args.dogmv
pathmov5 = args.ngmv
pathmov6 = args.resmv

outpath = args.outpath
pol = args.pol

paths={}

if args.kinemv!='':
    paths['kine']=args.kinemv
if args.starmv!='':
    paths['starwarps']=args.starmv
if args.ehtmv!='':
    paths['ehtim']=args.ehtmv
if args.dogmv!='':
    paths['doghit']=args.dogmv 
if args.ngmv!='':
    paths['ngmem']=args.ngmv
if args.resmv!='':
    paths['resolve']=args.resmv


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


    
polpaths={}
for p in paths.keys():
    mv=eh.movie.load_hdf5(paths[p])
    im=mv.get_image(times[0])
    
    if pol=='I':
       if len(im.ivec)>0:
        polpaths[p]=paths[p]
    elif pol=='Q' and p!='starwarps':
        if len(im.qvec)>0:
            polpaths[p]=paths[p]
    elif pol=='U' and p!='starwarps':
        if len(im.uvec)>0:
            polpaths[p]=paths[p]
    elif pol=='V' and p!='starwarps':
        if len(im.vvec)>0:
            polpaths[p]=paths[p]
    else:
        print('Parse a vaild pol value')


colors = {   
            'kine'     : 'darkorange',
            'starwarps': 'xkcd:azure',
            'ehtim'    : 'forestgreen',
            'doghit'   : 'darkviolet',
            'ngmem'    : 'crimson',
            'resolve'  : 'hotpink'
        }

labels = {  
            'kine'     : 'kine',
            'starwarps': 'StarWarps',
            'ehtim'    : 'ehtim',
            'doghit'   : 'DoG-HiT',
            'ngmem'    : 'ngMEM',
            'resolve'  : 'resolve'
        }

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
ax[0].text(10.5, 5e6, f'Stokes: {pol}', color='white', fontsize=18)

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
    c.set_text_props(color='white')
    c.set_facecolor('none')
    c.set_edgecolor('white')
        
plt.savefig(args.outpath+'.png', bbox_inches='tight', dpi=300)
