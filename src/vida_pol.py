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
from tqdm import tqdm
import itertools 
import sys
from copy import copy
from utilities import *
colors, titles, labels, mfcs, mss = common()

codedir = os.getcwd()

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
    p.add_argument('--scat', type=str, default='none', help='onsky, deblur, dsct, none')

    return p

# List of parsed arguments
args = create_parser().parse_args()

outpath = args.outpath
cores = args.cores
data = args.data
scat= args.data

paths={}
if args.truthmv!='none':
    paths['truth']=args.truthmv
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
obs, obs_t, obslist_t, splitObs, times, w_norm, equal_w = process_obs_weights(obs, args, paths)

######################################################################

for p in paths.keys():
    outpath_csv= outpath[:-4]+f'_{p}.csv'
    if not os.path.exists(outpath_csv):
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
                if args.scat!='onsky':
                    im = im.blur_circ(fwhm_i=15*eh.RADPERUAS, fwhm_pol=15*eh.RADPERUAS)
            im.save_fits(f"{folder}/temp/{times[i]}.fits")
            i=i+1
        
        os.system(f'realpath {folder}/temp/*.fits > {folder}/temp/filelist.txt')
        os.system(f'julia {codedir}/vida_pol.jl --imfiles {folder}/temp/filelist.txt --outname {outpath_csv} --stride {cores}')
        os.system(f'rm -r {folder}/temp/')

######################################################################

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
m_net_dict={}
m_avg_dict={}
ang_betalp_2_dict={}
ang_betacp_1_dict={}

for p in paths.keys():
    s[p]  = outpath[:-4]+f'_{p}.csv'
    
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
            
    ang_betacp_1 = ang_betacp_1%360
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

    m_net_dict[p] = m_net
    m_avg_dict[p] = m_avg
    ang_betalp_2_dict[p] = ang_betalp_2
    ang_betacp_1_dict[p] = ang_betacp_1

score={}
for p in paths.keys():
    if p!='truth':
        score[p]=np.zeros(4)

row_labels = ['$|m|_{net}$', '$ \\langle |m| \\rangle$', '$ \\angle \\beta_{LP,2}$', '$ \\angle \\beta_{CP,1}$']
table_vals = pd.DataFrame(data=score, index=row_labels)
 
for p in paths.keys():
    if p!='truth':
        if np.sum(m_net_dict[p])!=0:
            signal1 = m_net_dict['truth']
            signal2 = m_net_dict[p]
            table_vals[p][row_labels[0]] = normalized_rmse(signal1, signal2, w_norm['I'])

            signal1 = m_avg_dict['truth']
            signal2 = m_avg_dict[p]
            table_vals[p][row_labels[1]] = normalized_rmse(signal1, signal2, w_norm['I'])

            signal1 = ang_betalp_2_dict['truth']
            signal2 = ang_betalp_2_dict[p]
            table_vals[p][row_labels[2]] = normalized_rmse(signal1, signal2, w_norm['I'])

            signal1 = ang_betacp_1_dict['truth']
            signal2 = ang_betacp_1_dict[p]
            table_vals[p][row_labels[3]] = normalized_rmse(signal1, signal2, w_norm['I'])


table_vals.replace(0.00, '-', inplace=True)

table = ax[0,1].table(cellText=table_vals.values,
                    rowLabels=table_vals.index,
                    colLabels=table_vals.columns,
                    cellLoc='center',
                    loc='bottom',
                    bbox=[-0.1, -2.2, 3.5, 0.7])
table.auto_set_font_size(False)
table.set_fontsize(18)
for c in table.get_children():
    c.set_edgecolor('none')
    c.set_text_props(color='black')
    c.set_facecolor('none')
    c.set_edgecolor('black')
    
ax[0,0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(5.25, 1.35), markerscale=2)
plt.savefig(outpath, bbox_inches='tight', dpi=300)
