######################################################################
# Author: Rohan Dahale, Date: 12 July 2024
######################################################################

# Import libraries
import numpy as np
import pandas as pd
import ehtim as eh
import ehtim.scattering.stochastic_optics as so
from preimcal import *
import tqdm
import copy
import matplotlib.pyplot as plt
import pdb
import argparse
import os
import glob
from utilities import *
colors, titles, labels, mfcs, mss = common()

codedir = os.getcwd()


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
    p.add_argument('--model',    type=str, default='none', help='type of model: crescent, ring, disk, edisk, double, point, mring_1_4')
    p.add_argument('--template', type=str, default='none', help='VIDA template')
    p.add_argument('-c', '--cores', type=int, default='64',help='number of cores to use')
    p.add_argument('-o', '--outpath', type=str, default='./vida.png',
                   help='name of output file with path')
    p.add_argument('--scat', type=str, default='none', help='onsky, deblur, dsct, none')

    return p

# List of parsed arguments
args = create_parser().parse_args()
data = args.data
scat= args.data
outpath = args.outpath
cores = args.cores
model = args.model
template = args.template

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

outpath_csv={}

for p in paths.keys():
    outpath_csv[p]= outpath+f'_{p}.csv'
    if not os.path.exists(outpath_csv[p]):
        mv=eh.movie.load_hdf5(paths[p])
        iml=[mv.get_image(t) for t in times]
        
        new_movie = eh.movie.merge_im_list(iml)
        new_movie.reset_interp(bounds_error=False)

        file_path = args.outpath
        parts = file_path.rsplit('/', 1)

        if len(parts) == 2:
            folder, filename = parts
        os.system(f'mkdir -p {folder}/temp/')
        new_movie.save_hdf5(f"{folder}/temp/{os.path.basename(paths[p])}")

        input=f"{folder}/temp/{os.path.basename(paths[p])}"
        output = outpath_csv[p]
        if p=='truth':
            if args.scat!='onsky':
                os.system(f'julia -p {cores} {codedir}/movie_extractor_parallel.jl --input {input} --output {output} --template {template} --stride {cores} --blur 15.0')
        else:
            os.system(f'julia -p {cores} {codedir}/movie_extractor_parallel.jl --input {input} --output {output} --template {template} --stride {cores}')
        
        os.system(f'rm -r {folder}/temp/')
        
######################################################################
# Plots
######################################################################

if model=='crescent':
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,6), sharex=True)
    alpha = 1.0
    lc='grey'

    ax[0,0].set_ylabel('Diameter $d (\mu$as)')
    ax[0,0].set_ylim(35,65)

    ax[0,1].set_ylabel('$d(w) (\mu$as)')
    ax[0,1].set_ylim(35,65)

    ax[1,0].set_ylabel('PA '+ r'$\eta (^{\circ}$ E of N)')
    ax[1,0].set_ylim(-180,180)

    ax[1,1].set_ylabel('Bright. Asym. $A$')
    ax[1,1].set_ylim(0.0,0.6)


    ax[1,0].set_xlabel('Time (UTC)')
    ax[1,1].set_xlabel('Time (UTC)')


    for i in range(2):
        for j in range(2):
            ax[i,j].set_xlim(10.7,14.2)

    ######################################################################
    # Read the CSV with extracted parameters
    ######################################################################
    dw_dict={}
    n_dict={}
    a_dict={}
    
    for p in outpath_csv.keys():
        df = pd.read_csv(outpath_csv[p])
        d = 2*df['model_1_r0']/eh.RADPERUAS
        w0 = df['model_1_σ0']/eh.RADPERUAS*(2*np.sqrt(2*np.log(2)))

        for i in range(len(df['model_1_ξs_1'])):
            if df['model_1_ξs_1'][i]<-np.pi:
                df['model_1_ξs_1'][i] = df['model_1_ξs_1'][i] + 2*np.pi
            if df['model_1_ξs_1'][i]>np.pi:
                df['model_1_ξs_1'][i] = df['model_1_ξs_1'][i] - 2*np.pi

        a = df['model_1_s_1']/2
        n =np.rad2deg(df['model_1_ξs_1'])
        t = df['time']

        mc=colors[p]
        mfc=mfcs[p]
        ms=mss[p]
        
        true_d=np.array(d/(1-(1/(4*np.log(2)))*(w0/d)**2))
        
        ax[0,0].plot(t, d,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha, label=labels[p])
        ax[0,1].plot(t, true_d, marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,0].plot(t, n,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,1].plot(t, a,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha)

        dw_dict[p]=np.array(d/(1-(1/(4*np.log(2)))*(w0/d)**2))
        n_dict[p]=n
        a_dict[p]=a
        
    score={}
    for p in paths.keys():
        if p!='truth':
            score[p]=np.zeros(3)
    row_labels = ['$d(w)$', 'PA', '$A$']
    table_vals = pd.DataFrame(data=score, index=row_labels)
    for p in paths.keys():
        if p!='truth':
            signal1 = dw_dict['truth']
            signal2 = dw_dict[p]
            table_vals[p][row_labels[0]] = normalized_rmse(signal1, signal2, w_norm['I'])
            signal1 = n_dict['truth']
            signal2 = n_dict[p]
            table_vals[p][row_labels[1]] = normalized_rmse(signal1, signal2, w_norm['I'])
            signal1 = a_dict['truth']
            signal2 = a_dict[p]
            table_vals[p][row_labels[2]] = normalized_rmse(signal1, signal2, w_norm['I'])

    table_vals.replace(0.00, '-', inplace=True)

    table = ax[0,0].table(cellText=table_vals.values,
                        rowLabels=table_vals.index,
                        colLabels=table_vals.columns,
                        cellLoc='center',
                        loc='bottom',
                        bbox=[0.35, -2.3, 1.5, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    for c in table.get_children():
        c.set_edgecolor('none')
        c.set_text_props(color='black')
        c.set_facecolor('none')
        c.set_edgecolor('black')
    ax[0,0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(2.1, 1.4), markerscale=2.0)

elif model=='ring':
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,3), sharex=True)
     
    alpha = 0.5
    lc='grey'

    ax[0].set_ylabel('Diameter $d (\mu$as)')
    ax[0].set_ylim(35,80)
    ax[1].set_ylabel('width $w (\mu$as)')
    ax[1].set_ylim(10,35)
    ax[0].set_xlabel('Time (UTC)')
    ax[1].set_xlabel('Time (UTC)')


    for i in range(2):
        ax[i].set_xlim(10.7,14.2)

    ######################################################################
    # Read the CSV with extracted parameters
    ######################################################################
    d_dict={}
    w_dict={}
    dw_dict={}
    
    for p in outpath_csv.keys():
        df = pd.read_csv(outpath_csv[p])
        d = 2*df['model_1_r0']/eh.RADPERUAS
        w0 = df['model_1_σ0']/eh.RADPERUAS*(2*np.sqrt(2*np.log(2)))

        t = df['time']

        mc=colors[p]
        mfc=mfcs[p]
        ms=mss[p]
        
        ax[0].plot(t, d,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha, label=labels[p])
        ax[1].plot(t, w0, marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha)

        d_dict[p]=d
        w_dict[p]=w0
        dw_dict[p]=np.array(d/(1-(1/(4*np.log(2)))*(w0/d)**2))
        
    score={}
    for p in paths.keys():
        if p!='truth':
            score[p]=np.zeros(3)
    row_labels = ['$d$', '$w$', '$d(w)$']
    table_vals = pd.DataFrame(data=score, index=row_labels)
    for p in paths.keys():
        if p!='truth':
            signal1 = d_dict['truth']
            signal2 = d_dict[p]
            table_vals[p][row_labels[0]] = normalized_rmse(signal1, signal2, w_norm['I'])
            signal1 = w_dict['truth']
            signal2 = w_dict[p]
            table_vals[p][row_labels[1]] = normalized_rmse(signal1, signal2, w_norm['I'])
            signal1 = dw_dict['truth']
            signal2 = dw_dict[p]
            table_vals[p][row_labels[2]] = normalized_rmse(signal1, signal2, w_norm['I'])

    table_vals.replace(0.00, '-', inplace=True)

    table = ax[0].table(cellText=table_vals.values,
                        rowLabels=table_vals.index,
                        colLabels=table_vals.columns,
                        cellLoc='center',
                        loc='bottom',
                        bbox=[0.35, -0.9, 1.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    for c in table.get_children():
        c.set_edgecolor('none')
        c.set_text_props(color='black')
        c.set_facecolor('none')
        c.set_edgecolor('black')
        
    ax[0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(2.1, 1.4), markerscale=2.0)

elif model=='disk':
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,3), sharex=True)
     
    alpha = 0.5
    lc='grey'

    ax[0].set_ylabel('Diameter $d (\mu$as)')
    ax[0].set_ylim(35,80)
    ax[1].set_ylabel('Gauss FWHM ($\mu$as)')
    ax[1].set_ylim(10,35)
    ax[0].set_xlabel('Time (UTC)')
    ax[1].set_xlabel('Time (UTC)')


    for i in range(2):
        ax[i].set_xlim(10.7,14.2)

    ######################################################################
    # Read the CSV with extracted parameters
    ######################################################################
    d_dict={}
    w_dict={}
    
    for p in outpath_csv.keys():
        df = pd.read_csv(outpath_csv[p])
        d = 2*df['model_1_r0_1']/eh.RADPERUAS
        w0 = df['model_1_σ_1']/eh.RADPERUAS*(2*np.sqrt(2*np.log(2)))

        t = df['time']

        mc=colors[p]
        mfc=mfcs[p]
        ms=mss[p]
        ax[0].plot(t, d,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha, label=labels[p])
        ax[1].plot(t, w0, marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha)

        d_dict[p]=d
        w_dict[p]=w0
        
    score={}
    for p in paths.keys():
        if p!='truth':
            score[p]=np.zeros(2)
    row_labels = ['$d$', '$w$']
    table_vals = pd.DataFrame(data=score, index=row_labels)
    for p in paths.keys():
        if p!='truth':
            signal1 = d_dict['truth']
            signal2 = d_dict[p]
            table_vals[p][row_labels[0]] = normalized_rmse(signal1, signal2, w_norm['I'])
            signal1 = w_dict['truth']
            signal2 = w_dict[p]
            table_vals[p][row_labels[1]] = normalized_rmse(signal1, signal2, w_norm['I'])
    
    table_vals.replace(0.00, '-', inplace=True)

    table = ax[0,0].table(cellText=table_vals.values,
                        rowLabels=table_vals.index,
                        colLabels=table_vals.columns,
                        cellLoc='center',
                        loc='bottom',
                        bbox=[0.35, -0.9, 1.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    for c in table.get_children():
        c.set_edgecolor('none')
        c.set_text_props(color='black')
        c.set_facecolor('none')
        c.set_edgecolor('black')
        
    ax[0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(2.1, 1.4), markerscale=2.0)

elif model=='edisk':
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,6), sharex=True)
    alpha = 0.5
    lc='grey'

    ax[0,0].set_ylabel('Diameter $d (\mu$as)')
    ax[0,0].set_ylim(35,80)

    ax[0,1].set_ylabel('Guass FWHM ($\mu$as)')
    ax[0,1].set_ylim(10,35)

    ax[1,0].set_ylabel('Ellipticity '+r'$\tau$')
    ax[1,0].set_ylim(0.0,0.3)
    
    ax[1,1].set_ylabel(r'$\xi_\tau (^{\circ}$ E of N)')
    ax[1,1].set_ylim(-90,90)

    ax[1,0].set_xlabel('Time (UTC)')
    ax[1,1].set_xlabel('Time (UTC)')


    for i in range(2):
        for j in range(2):
            ax[i,j].set_xlim(10.7,14.2)

    ######################################################################
    # Read the CSV with extracted parameters
    ######################################################################
    d_dict={}
    w_dict={}
    e_dict={}
    n_dict={}
    
    for p in outpath_csv.keys():
        df = pd.read_csv(outpath_csv[p])
        d = 2*df['model_1_r0_1']/eh.RADPERUAS
        w0 = df['model_1_σ_1']/eh.RADPERUAS*(2*np.sqrt(2*np.log(2)))
        e = abs(df['model_1_τ_1'])

        for i in range(len(df['model_1_ξ_1'])):
            if df['model_1_ξ_1'][i]<-np.pi/2:
                df['model_1_ξ_1'][i] = df['model_1_ξ_1'][i] + np.pi
            if df['model_1_ξ_1'][i]>np.pi/2:
                df['model_1_ξ_1'][i] = df['model_1_ξ_1'][i] - np.pi

        n =np.rad2deg(df['model_1_ξ_1'])
        t = df['time']

        mc=colors[p]
        mfc=mfcs[p]
        ms=mss[p]
        ax[0,0].plot(t, d,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha, label=labels[p])
        ax[0,1].plot(t, w0, marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,0].plot(t, e,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,1].plot(t, n,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha)

        d_dict[p]=d
        w_dict[p]=w0
        e_dict[p]=e
        n_dict[p]=n
        
    score={}
    for p in paths.keys():
        if p!='truth':
            score[p]=np.zeros(4)
    row_labels = ['$d$', '$w$', '$\tau$', 'PA']
    table_vals = pd.DataFrame(data=score, index=row_labels)
    for p in paths.keys():
        if p!='truth':
            signal1 = d_dict['truth']
            signal2 = d_dict[p]
            table_vals[p][row_labels[0]] = normalized_rmse(signal1, signal2, w_norm['I'])
            signal1 = w_dict['truth']
            signal2 = w_dict[p]
            table_vals[p][row_labels[1]] = normalized_rmse(signal1, signal2, w_norm['I'])
            signal1 = e_dict['truth']
            signal2 = e_dict[p]
            table_vals[p][row_labels[2]] = normalized_rmse(signal1, signal2, w_norm['I'])
            signal1 = n_dict['truth']
            signal2 = n_dict[p]
            table_vals[p][row_labels[3]] = normalized_rmse(signal1, signal2, w_norm['I'])
    
    table_vals.replace(0.00, '-', inplace=True)

    table = ax[0,0].table(cellText=table_vals.values,
                        rowLabels=table_vals.index,
                        colLabels=table_vals.columns,
                        cellLoc='center',
                        loc='bottom',
                        bbox=[0.35, -2.3, 1.5, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    for c in table.get_children():
        c.set_edgecolor('none')
        c.set_text_props(color='black')
        c.set_facecolor('none')
        c.set_edgecolor('black')
        
    ax[0,0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(2.1, 1.4), markerscale=2.0)

elif model=='double':
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,6), sharex=True)
    alpha = 0.5
    lc='grey'

    ax[0,0].set_ylabel('FWHM 1 ($\mu$as)')
    ax[0,0].set_ylim(0,60)

    ax[0,1].set_ylabel('FWHM 2 ($\mu$as)')
    ax[0,1].set_ylim(0,60)

    ax[1,0].set_ylabel('Separation')
    ax[1,0].set_ylim(40,80)
    
    ax[1,1].set_ylabel(r'PA ($^{\circ}$ E of N)')
    ax[1,1].set_ylim(-180,180)

    ax[1,0].set_xlabel('Time (UTC)')
    ax[1,1].set_xlabel('Time (UTC)')


    for i in range(2):
        for j in range(2):
            ax[i,j].set_xlim(10.7,14.2)

    ######################################################################
    # Read the CSV with extracted parameters
    ######################################################################
    d1_dict={}
    d2_dict={}
    r0_dict={}
    pa_dict={}
    
    for p in outpath_csv.keys():
        df = pd.read_csv(outpath_csv[p])
        
        d1 = np.array(df['model_1_σ_1']/eh.RADPERUAS*(2*np.sqrt(2*np.log(2))))
        d2 = np.array(df['model_1_σ_2']/eh.RADPERUAS*(2*np.sqrt(2*np.log(2))))
        x01 = df['model_1_x0_1']/eh.RADPERUAS
        y01 = df['model_1_y0_1']/eh.RADPERUAS
        x02 = df['model_1_x0_2']/eh.RADPERUAS
        y02 = df['model_1_y0_2']/eh.RADPERUAS
        
        for k in range(len(d1)):
            if d1[k] < d2[k]:
                d1[k] = d2[k]
                d2[k] = d1[k]
                
                x01[k] = x02[k]
                x02[k] = x01[k]
                y01[k] = y02[k]
                y01[k] = y01[k]
        
        pos = np.abs(df['model_1_x0_1']-df['model_1_x0_2'])/eh.RADPERUAS + 1j*np.abs(df['model_1_y0_1']-df['model_1_y0_2'])/eh.RADPERUAS
        r0 = np.abs(pos)
        pa = np.rad2deg(np.angle(pos))
        t = df['time']

        mc=colors[p]
        mfc=mfcs[p]
        ms=mss[p]
        ax[0,0].plot(t, d1,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha, label=labels[p])
        ax[0,1].plot(t, d2,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,0].plot(t, r0,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,1].plot(t, pa,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha)

        d1_dict[p]=d1
        d2_dict[p]=d2
        r0_dict[p]=r0
        pa_dict[p]=pa
        
    score={}
    for p in paths.keys():
        if p!='truth':
            score[p]=np.zeros(4)
    row_labels = ['FWHM1', 'FWHM2', 'Separation', 'PA']
    table_vals = pd.DataFrame(data=score, index=row_labels)
    for p in paths.keys():
        if p!='truth':
            signal1 = d1_dict['truth']
            signal2 = d1_dict[p]
            table_vals[p][row_labels[0]] = normalized_rmse(signal1, signal2, w_norm['I'])
            signal1 = d2_dict['truth']
            signal2 = d2_dict[p]
            table_vals[p][row_labels[1]] = normalized_rmse(signal1, signal2, w_norm['I'])
            signal1 = r0_dict['truth']
            signal2 = r0_dict[p]
            table_vals[p][row_labels[2]] = normalized_rmse(signal1, signal2, w_norm['I'])
            signal1 = pa_dict['truth']
            signal2 = pa_dict[p]
            table_vals[p][row_labels[3]] = normalized_rmse(signal1, signal2, w_norm['I'])
    
    table_vals.replace(0.00, '-', inplace=True)

    table = ax[0,0].table(cellText=table_vals.values,
                        rowLabels=table_vals.index,
                        colLabels=table_vals.columns,
                        cellLoc='center',
                        loc='bottom',
                        bbox=[0.35, -2.3, 1.5, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    for c in table.get_children():
        c.set_edgecolor('none')
        c.set_text_props(color='black')
        c.set_facecolor('none')
        c.set_edgecolor('black')
        
    ax[0,0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(2.1, 1.4), markerscale=2.0)

elif model=='point':
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,6), sharex=True)
    alpha = 0.5
    lc='grey'

    ax[0,0].set_ylabel('FWHM 1 ($\mu$as)')
    ax[0,0].set_ylim(0,120)

    ax[0,1].set_ylabel('FWHM 2 ($\mu$as)')
    ax[0,1].set_ylim(0,120)

    ax[1,0].set_ylabel('Separation')
    ax[1,0].set_ylim(-10,50)
    
    ax[1,1].set_ylabel(r'PA ($^{\circ}$ E of N)')
    ax[1,1].set_ylim(-180,180)

    ax[1,0].set_xlabel('Time (UTC)')
    ax[1,1].set_xlabel('Time (UTC)')


    for i in range(2):
        for j in range(2):
            ax[i,j].set_xlim(10.7,14.2)

    ######################################################################
    # Read the CSV with extracted parameters
    ######################################################################
    d1_dict={}
    d2_dict={}
    r0_dict={}
    pa_dict={}
    
    for p in outpath_csv.keys():
        df = pd.read_csv(outpath_csv[p])
        
        d1 = np.array(df['model_1_σ_1']/eh.RADPERUAS*(2*np.sqrt(2*np.log(2))))
        d2 = np.array(df['model_1_σ_2']/eh.RADPERUAS*(2*np.sqrt(2*np.log(2))))
        x01 = df['model_1_x0_1']/eh.RADPERUAS
        y01 = df['model_1_y0_1']/eh.RADPERUAS
        x02 = df['model_1_x0_2']/eh.RADPERUAS
        y02 = df['model_1_y0_2']/eh.RADPERUAS
        
        for k in range(len(d1)):
            if d1[k] < d2[k]:
                d1[k] = d2[k]
                d2[k] = d1[k]
                
                x01[k] = x02[k]
                x02[k] = x01[k]
                y01[k] = y02[k]
                y01[k] = y01[k]
        
        pos = np.abs(df['model_1_x0_1']-df['model_1_x0_2'])/eh.RADPERUAS + 1j*np.abs(df['model_1_y0_1']-df['model_1_y0_2'])/eh.RADPERUAS
        r0 = np.abs(pos)
        pa = np.rad2deg(np.angle(pos))
        t = df['time']

        mc=colors[p]
        mfc=mfcs[p]
        ms=mss[p]
        ax[0,0].plot(t, d1,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha, label=labels[p])
        ax[0,1].plot(t, d2,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,0].plot(t, r0,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,1].plot(t, pa,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha)

        d1_dict[p]=d1
        d2_dict[p]=d2
        r0_dict[p]=r0
        pa_dict[p]=pa
        
    score={}
    for p in paths.keys():
        if p!='truth':
            score[p]=np.zeros(4)
    row_labels = ['FWHM1', 'FWHM2', 'Separation', 'PA']
    table_vals = pd.DataFrame(data=score, index=row_labels)
    for p in paths.keys():
        if p!='truth':
            signal1 = d1_dict['truth']
            signal2 = d1_dict[p]
            table_vals[p][row_labels[0]] = normalized_rmse(signal1, signal2, w_norm['I'])
            signal1 = d2_dict['truth']
            signal2 = d2_dict[p]
            table_vals[p][row_labels[1]] = normalized_rmse(signal1, signal2, w_norm['I'])
            signal1 = r0_dict['truth']
            signal2 = r0_dict[p]
            table_vals[p][row_labels[2]] = normalized_rmse(signal1, signal2, w_norm['I'])
            signal1 = pa_dict['truth']
            signal2 = pa_dict[p]
            table_vals[p][row_labels[3]] = normalized_rmse(signal1, signal2, w_norm['I'])
    
    table_vals.replace(0.00, '-', inplace=True)

    table = ax[0,0].table(cellText=table_vals.values,
                        rowLabels=table_vals.index,
                        colLabels=table_vals.columns,
                        cellLoc='center',
                        loc='bottom',
                        bbox=[0.35, -2.3, 1.5, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    for c in table.get_children():
        c.set_edgecolor('none')
        c.set_text_props(color='black')
        c.set_facecolor('none')
        c.set_edgecolor('black')
        
    ax[0,0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(2.1, 1.4), markerscale=2.0)

elif model=='gaussian':
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,6), sharex=True)
    alpha = 0.5
    lc='grey'

    ax[0,0].set_ylabel('FWHM ($\mu$as)')
    ax[0,0].set_ylim(0,120)

    ax[0,1].set_ylabel('$x (\mu$as)')
    ax[0,1].set_ylim(-60,60)

    ax[1,0].set_ylabel('Distance from 0,0')
    ax[1,0].set_ylim(-10,50)
    
    ax[1,1].set_ylabel(r'PA ($^{\circ}$)')
    ax[1,1].set_ylim(-180,180)

    ax[1,0].set_xlabel('Time (UTC)')
    ax[1,1].set_xlabel('Time (UTC)')


    for i in range(2):
        for j in range(2):
            ax[i,j].set_xlim(10.7,14.2)

    ######################################################################
    # Read the CSV with extracted parameters
    ######################################################################
    d1_dict={}
    r0_dict={}
    pa_dict={}
    
    for p in outpath_csv.keys():
        df = pd.read_csv(outpath_csv[p])
        
        d1 = np.array(df['model_1_σ_1']/eh.RADPERUAS*(2*np.sqrt(2*np.log(2))))
        x01 = df['model_1_x0_1']/eh.RADPERUAS
        y01 = df['model_1_y0_1']/eh.RADPERUAS
        
        pos = df['model_1_x0_1']/eh.RADPERUAS + 1j*df['model_1_y0_1']/eh.RADPERUAS
        r0 = np.abs(pos)
        pa = np.rad2deg(np.angle(pos))
        t = df['time']

        mc=colors[p]
        mfc=mfcs[p]
        ms=mss[p]
        ax[0,0].plot(t, d1,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha, label=labels[p])
        ax[0,1].plot(t, x01, marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,0].plot(t, r0,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha)
        ax[1,1].plot(t, pa,  marker ='o', mfc=mfc, mec=mc, ms=ms, ls='-', lw=1, color=lc, alpha=alpha)

        d1_dict[p]=d1
        r0_dict[p]=r0
        pa_dict[p]=pa
        
    score={}
    for p in paths.keys():
        if p!='truth':
            score[p]=np.zeros(3)
    row_labels = ['FWHM', 'Distance', 'PA']
    table_vals = pd.DataFrame(data=score, index=row_labels)
    for p in paths.keys():
        if p!='truth':
            signal1 = d1_dict['truth']
            signal2 = d1_dict[p]
            table_vals[p][row_labels[0]] = normalized_rmse(signal1, signal2, w_norm['I'])
            signal1 = r0_dict['truth']
            signal2 = r0_dict[p]
            table_vals[p][row_labels[1]] = normalized_rmse(signal1, signal2, w_norm['I'])
            signal1 = pa_dict['truth']
            signal2 = pa_dict[p]
            table_vals[p][row_labels[2]] = normalized_rmse(signal1, signal2, w_norm['I'])
    
    table_vals.replace(0.00, '-', inplace=True)

    table = ax[0,0].table(cellText=table_vals.values,
                        rowLabels=table_vals.index,
                        colLabels=table_vals.columns,
                        cellLoc='center',
                        loc='bottom',
                        bbox=[0.35, -2.3, 1.5, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    for c in table.get_children():
        c.set_edgecolor('none')
        c.set_text_props(color='black')
        c.set_facecolor('none')
        c.set_edgecolor('black')
        
    ax[0,0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(2.1, 1.4), markerscale=2.0)
    
else:
    print('Model not in the list of plot functions')
    
plt.savefig(outpath+'.png', bbox_inches='tight', dpi=300)
print(f'{os.path.basename(outpath)} is created')