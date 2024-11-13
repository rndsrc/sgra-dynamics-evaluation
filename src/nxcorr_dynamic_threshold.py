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
import scipy
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
    p.add_argument('--kinemv',  type=str, default='none', help='path of kine .hdf5')
    p.add_argument('--ehtmv',   type=str, default='none', help='path of ehtim .hdf5')
    p.add_argument('--dogmv',   type=str, default='none', help='path of doghit .hdf5')
    p.add_argument('--ngmv',    type=str, default='none', help='path of ngmem .hdf5')
    p.add_argument('--resmv',   type=str, default='none', help='path of resolve .hdf5')
    p.add_argument('--modelingmv',  type=str, default='none', help='path of modeling .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./chi2.png', 
                   help='name of output file with path')
    p.add_argument('--scat', type=str, default='none', help='onsky, deblur, dsct, none')
    return p

def normalized_cross_correlation(matrix1, matrix2):
    # Ensure the matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same shape")

    # Subtract the mean from the matrices
    matrix1_mean_subtracted = matrix1 - np.mean(matrix1)
    matrix2_mean_subtracted = matrix2 - np.mean(matrix2)

    # Compute the numerator of the normalized cross-correlation
    numerator = np.sum(matrix1_mean_subtracted * matrix2_mean_subtracted)

    # Compute the denominator of the normalized cross-correlation
    denominator = np.sqrt(np.sum(matrix1_mean_subtracted ** 2) * np.sum(matrix2_mean_subtracted ** 2))

    # Compute the normalized cross-correlation
    ncc = numerator / denominator

    return ncc


# List of parsed arguments
args = create_parser().parse_args()

pathmovt  = args.truthmv
outpath = args.outpath

npix   = 200
fov    = 200 * eh.RADPERUAS

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
if args.modelingmv!='none':
    paths['modeling']=args.modelingmv
    
#################################################################################

obs = eh.obsdata.load_uvfits(args.data)
obs, obs_t, obslist_t, splitObs, times, I, snr, w_norm = process_obs_weights(obs, args, paths)

######################################################################

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21,6), sharex=True)

ax[0].set_ylabel('$\Delta$ nxcorr (I)')
ax[1].set_ylabel('$\Delta$ nxcorr (Q)')
ax[2].set_ylabel('$\Delta$ nxcorr (U)')
ax#[3].set_ylabel('$\Delta$ nxcorr (V)')

ax[0].set_xlabel('Time (UTC)')
ax[1].set_xlabel('Time (UTC)')
ax[2].set_xlabel('Time (UTC)')
#ax[3].set_xlabel('Time (UTC)')

ax[0].set_ylim(-1.1,1.1)
ax[1].set_ylim(-1.1,1.1)
ax[2].set_ylim(-1.1,1.1)
#ax[3].set_ylim(-1.1,1.1)

#ax[0].set_ylim(0,7)
#ax[1].set_ylim(0,7)
#ax[2].set_ylim(0,7)
#ax[3].set_ylim(0,7)

mvt=eh.movie.load_hdf5(pathmovt)
if args.scat!='onsky':
    mvt=mvt.blur_circ(fwhm_x=15*eh.RADPERUAS, fwhm_x_pol=15*eh.RADPERUAS, fwhm_t=0)
    
mvt_list= mvt.im_list()
mvt_list2=[]
for im in mvt_list:
    im = im.regrid_image(fov, npix)
    mvt_list2.append(im)
mvt=eh.movie.merge_im_list(mvt_list2)

mv_nxcorr={}
for p in paths.keys():
    mv_nxcorr[p]=np.zeros(3)

row_labels = ['I','Q','U']
table_vals = pd.DataFrame(data=mv_nxcorr, index=row_labels)
        
pollist=['I','Q','U']
k=0
for pol in pollist:
    polpaths={}
    for p in paths.keys():
        mv=eh.movie.load_hdf5(paths[p])
        im=mv.im_list()[0]
        
        if pol=='I':
            if len(im.ivec)>0:
                polpaths[p]=paths[p]
            else:
                print('Parse a vaild pol value')
        elif pol=='Q':
            if len(im.qvec)>0:
                polpaths[p]=paths[p]
            else:
                print('Parse a vaild pol value')
        elif pol=='U':
            if len(im.uvec)>0:
                polpaths[p]=paths[p]
            else:
                print('Parse a vaild pol value')
        elif pol=='V':
            if len(im.vvec)>0:
                polpaths[p]=paths[p]
            else:
                print('Parse a vaild pol value')
        else:
            print('Parse a vaild pol value')

    s=0
    for p in polpaths.keys():
        mv=eh.movie.load_hdf5(polpaths[p])
        
        mv_list= mv.im_list()
        mv_list2=[]
        for im in mv_list:
            im = im.regrid_image(fov, npix)
            mv_list2.append(im)
        mv=eh.movie.merge_im_list(mv_list2)

        imlist = [mv.get_image(t) for t in times]
        imlist_t = [mvt.get_image(t) for t in times]
        
        """
        # center the movie with repect to the truth movie frame 0
        mvtruth_image=eh.movie.load_hdf5(pathmovt).avg_frame()
        shift = mvtruth_image.align_images([mv.avg_frame()])[1] #Shifts only from stokes I
        imlist=[]
        for im in imlist_o:
            im2 = im.shift(shift[0]) # Shifts all pol
            imlist.append(im2)
        """
        
        # Shifting each frame of the movie to align with the truth
        imlistarr=[]
        imlist_aligned=[]
        for im, imt in zip(imlist, imlist_t):
            shift = imt.align_images([im])[1]
            im = im.shift(shift[0])
            imlist_aligned.append(im)
            imlistarr.append(im.imarr(pol=pol))
            
        # Subtracting the median of the reconstructed video from each frame    
        median = np.median(imlistarr,axis=0)
        for im in imlist_aligned:
            if pol=='I':
                im.ivec= np.array(im.imarr(pol=pol)-median).flatten()
            elif pol=='Q':
                im.qvec= np.array(im.imarr(pol=pol)-median).flatten()
            elif pol=='U':
                im.uvec= np.array(im.imarr(pol=pol)-median).flatten()
            elif pol=='V':
                im.vvec= np.array(im.imarr(pol=pol)-median).flatten()
    
        # Subtracting the median of the truth video from each frame
        imlistarr=[]
        for im in imlist_t:
            imlistarr.append(im.imarr(pol=pol))
        
        median = np.median(imlistarr,axis=0)
        for im in imlist_t:
            if pol=='I':
                im.ivec= np.array(im.imarr(pol=pol)-median).flatten()
            elif pol=='Q':
                im.qvec= np.array(im.imarr(pol=pol)-median).flatten()
            elif pol=='U':
                im.uvec= np.array(im.imarr(pol=pol)-median).flatten()
            elif pol=='V':
                im.vvec= np.array(im.imarr(pol=pol)-median).flatten()

        nxcorr_t=[]
        nxs_cri=[]
        nxcorr_tab=[]

        i=0
        for im in imlist_aligned:
            im = im.regrid_image(160*eh.RADPERUAS, 32)
            imlist_t[i] = imlist_t[i].regrid_image(160*eh.RADPERUAS, 32)
            
            if pol=='I':
                nxcorr = normalized_cross_correlation(imlist_t[i].ivec.reshape(32,32), im.ivec.reshape(32,32))
            elif pol=='Q':
                nxcorr = normalized_cross_correlation(imlist_t[i].qvec.reshape(32,32), im.qvec.reshape(32,32))
            elif pol=='U':
                nxcorr = normalized_cross_correlation(imlist_t[i].uvec.reshape(32,32), im.uvec.reshape(32,32))
            else:
                nxcorr = normalized_cross_correlation(imlist_t[i].vvec.reshape(32,32), im.vvec.reshape(32,32))
        
            nxcorr_t.append(nxcorr)
            nxcorr_tab.append(nxcorr)
            beamparams = obslist_t[i].fit_beam(weighting='uniform')
            nxs_cri.append(get_nxcorr_cri_beam(imlist_t[i], beamparams, pol=pol))
            i=i+1
        
        w_ratio = w_norm[pol]/np.max(w_norm[pol])
        w_ncri  = w_ratio*np.array(nxs_cri)
        
        count=0
        diff=np.array(nxcorr_t)-np.array(w_ncri)
        #diff=np.array(nxcorr_t)-np.array(nxs_cri)
        for d in diff:
            if d>0:
                count=count+1
        #if count==0:
        #    table_vals[p][pol]=1000
        #else:
        #    table_vals[p][pol]=np.round(count/len(nxcorr_t)*100,2)
        table_vals[p][pol]=np.round(count/len(nxcorr_t)*100,2)
                    
        mc=colors[p]
        alpha=1.0
        lc=colors[p]
        
        if k==0:
            ax[k].plot(times, nxcorr_t-w_ncri,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1,  color=lc, alpha=alpha, label=labels[p])
            #ax[k].plot(times, np.array(nxcorr_t)-np.array(nxs_cri),  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1,  color=lc, alpha=alpha, label=labels[p])
            #ax[k].plot(times, w_ncri, ls='--', lw=1,  color='k', label='Threshold')
        else:
            ax[k].plot(times, nxcorr_t-w_ncri,  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1,  color=lc, alpha=alpha)
            #ax[k].plot(times, np.array(nxcorr_t)-np.array(nxs_cri),  marker ='o', mfc=mc, mec=mc, mew=2.5, ms=2.5, ls='-', lw=1,  color=lc, alpha=alpha)
            #ax[k].plot(times, w_ncri, ls='--', lw=1,  color='k')
    
        #ax[k].hlines(s+1, xmin=10.5, xmax=14.5, color=colors[p], ls='--', lw=1.5, zorder=0)
        ax[k].hlines(0, xmin=10.5, xmax=14.5, color='k', ls='--', lw=1.5, zorder=0)
        #ax[k].yaxis.set_ticklabels([])
        s=s+1
    
    k=k+1
    
table_vals.rename(index={'I':'% Pass (I)'},inplace=True)
table_vals.rename(index={'Q':'% Pass (Q)'},inplace=True)
table_vals.rename(index={'U':'% Pass (U)'},inplace=True)
#table_vals.rename(index={'V':'% Pass (V)'},inplace=True)
#table_vals.replace(1000, 'Fail', inplace=True)
#table_vals.replace(0.000, '-', inplace=True)

col_labels=[]
for p in table_vals.keys():
    col_labels.append(titles[p])
    
table = ax[1].table(cellText=table_vals.values,
                    rowLabels=table_vals.index,
                    colLabels=col_labels,#table_vals.columns,
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
ax[0].legend(ncols=len(paths.keys()), loc='best',  bbox_to_anchor=(3.5, 1.2), markerscale=5.0)
plt.savefig(args.outpath+'.png', bbox_inches='tight', dpi=300)
