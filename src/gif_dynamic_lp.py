######################################################################
# Author: Rohan Dahale, Date: 12 July 2024
######################################################################

import ehtim as eh
import ehtim.scattering.stochastic_optics as so
from preimcal import *
import ehtplot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pdb
import argparse
import os
import glob
from utilities import *
colors, titles, labels, mfcs, mss = common()
plt.rcParams["xtick.direction"]="out"
plt.rcParams["ytick.direction"]="out"

# Parsing arguments function
def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data', type=str, 
                   default='hops_3601_SGRA_LO_netcal_LMTcal_10s_ALMArot_dcal.uvfits', 
                   help='string of uvfits to data to compute chi2')
    p.add_argument('--truthmv', type=str, default='none', help='path of truth .hdf5')
    p.add_argument('--kinemv', type=str, default='none', help='path of kine .hdf5')
    p.add_argument('--starmv', type=str, default='none', help='path of starwarps .hdf5')
    p.add_argument('--ehtmv',  type=str, default='none', help='path of ehtim .hdf5')
    p.add_argument('--dogmv',  type=str, default='none', help='path of doghit .hdf5')
    p.add_argument('--ngmv',   type=str, default='none', help='path of ngmem .hdf5')
    p.add_argument('--resmv',  type=str, default='none',help='path of resolve .hdf5')
    p.add_argument('-o', '--outpath', type=str, default='./gif.gif', 
                   help='name of output file with path')
    p.add_argument('--scat', type=str, default='none', help='onsky, deblur, dsct, none')

    return p

# List of parsed arguments
args = create_parser().parse_args()

outpath = args.outpath

paths={}

if args.truthmv!='none':
    paths['truth']=args.truthmv
if args.kinemv!='none':
    paths['kine']=args.kinemv
if args.resmv!='none':
    paths['resolve']=args.resmv
if args.starmv!='none':
    paths['starwarps']=args.starmv
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
# Set parameters
npix   = 128
fov    = 120 * eh.RADPERUAS
blur   = 0 * eh.RADPERUAS
######################################################################

# Adding times where there are gaps and assigning cmap as binary_us in the gaps
dt=[]
for i in range(len(times)-1):
    dt.append(times[i+1]-times[i])
    
mean_dt=np.mean(np.array(dt))

u_times=[]
cmapsl = []
for i in range(len(times)-1):
    if times[i+1]-times[i] > mean_dt:
        j=0
        while u_times[len(u_times)-1] < times[i+1]-mean_dt:
            u_times.append(times[i]+j*mean_dt)
            cmapsl.append('binary')
            j=j+1
    else:
        u_times.append(times[i])
        cmapsl.append('binary')

######################################################################

imlistIs = {}
for p in paths.keys():
    mov = eh.movie.load_hdf5(paths[p])
    imlistI = []
    imlistarrI = []
    imlistarrQ = []
    imlistarrU = []
    #imlistarrm = []
    for t in u_times:
        im = mov.get_image(t)
        #if p=='truth':
        #    if args.scat !='onsky':
        #        im = im.blur_circ(fwhm_i=15*eh.RADPERUAS, fwhm_pol=15*eh.RADPERUAS).regrid_image(fov, npix)
        im = im.blur_circ(fwhm_i=blur, fwhm_pol=blur).regrid_image(fov, npix)
        imlistI.append(im)
        imlistarrI.append(im.imarr(pol='I'))
        
        #if len(im.qvec) and len(im.uvec) > 0 and p!='starwarps':
        #    mfrac_arr = np.sqrt(im.imarr(pol='Q')**2 + im.imarr(pol='U')**2)/im.imarr(pol='I')
        #    imlistarrm.append(mfrac_arr)
        
        if len(im.qvec) and len(im.uvec) > 0 and p!='starwarps':
            imlistarrQ.append(im.imarr(pol='Q'))
            imlistarrU.append(im.imarr(pol='U'))
            
    medianI = np.median(imlistarrI,axis=0)
    if len(imlistarrQ) and len(imlistarrQ) > 0:
        medianQ = np.median(imlistarrQ,axis=0)
        medianU = np.median(imlistarrU,axis=0)
    #if len(imlistarrm)>0:
    #    medianm = np.median(imlistarrm,axis=0).flatten()

    for im in imlistI:
        #if len(im.qvec) and len(im.uvec) > 0 and p!='starwarps':
        #    frac_arr = np.array(np.sqrt(im.imarr(pol='Q')**2 + im.imarr(pol='U')**2)/im.imarr(pol='I')).flatten()
        im.ivec= np.array(im.imarr(pol='I')-medianI).flatten()
        if len(im.qvec) and len(im.uvec) > 0 and p!='starwarps':
            im.qvec= np.array(im.imarr(pol='Q')-medianQ).flatten()
            im.uvec= np.array(im.imarr(pol='U')-medianU).flatten()
        #if len(im.qvec) and len(im.uvec) > 0 and p!='starwarps':
        #    q_arr= im.qvec - (im.qvec*medianm/frac_arr)
        #    u_arr= im.uvec - (im.uvec*medianm/frac_arr)
        #    
        #    im.qvec= np.array(q_arr)
        #    im.uvec= np.array(u_arr)
            
            
    imlistIs[p] =imlistI

def linear_interpolation(x, x1, y1, x2, y2):
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

def writegif(movieIs, titles, paths, outpath='./', fov=None, times=[], cmaps=cmapsl, interp='gaussian', fps=20):
    num_subplots=len(paths.keys())
    fig, ax = plt.subplots(nrows=1, ncols=len(paths.keys()), figsize=(linear_interpolation(num_subplots, 2, 8, 7, 16),linear_interpolation(num_subplots, 2, 4, 7, 3)))
    #fig.tight_layout()
    fig.subplots_adjust(hspace=linear_interpolation(num_subplots, 2, 0.01, 7, 0.1), wspace=linear_interpolation(num_subplots, 2, 0.05, 7, 0.1), top=linear_interpolation(num_subplots, 2, 0.8, 7, 0.7), bottom=linear_interpolation(num_subplots, 2, 0.01, 7, 0.1), left=linear_interpolation(num_subplots, 2, 0.01, 7, 0.005), right=linear_interpolation(num_subplots, 2, 0.8, 7, 0.9))

    # Set axis limits
    lims = None
    if fov:
        fov  = fov / eh.RADPERUAS
        lims = [fov//2, -fov//2, -fov//2, fov//2]

    # Set colorbar limits
    TBfactor = 3.254e13/(movieIs[list(paths.keys())[0]][0].rf**2 * movieIs[list(paths.keys())[0]][0].psize**2)/1e9    
    vmax, vmin = max(movieIs[list(paths.keys())[0]][0].ivec)*TBfactor, 0 #min(movieIs['kine'][0].ivec)*TBfactor
    
    polmovies={}
    for i, p in enumerate(movieIs.keys()):    
        if len(movieIs[p][0].qvec) and len(movieIs[p][0].uvec) > 0 and p!='starwarps':
            polmovies[p]=movieIs[p]
                
    def plot_frame(f):
        for i, p in enumerate(movieIs.keys()):
            ax[i].clear() 
            TBfactor = 3.254e13/(movieIs[p][f].rf**2 * movieIs[p][f].psize**2)/1e9
            im =ax[i].imshow(np.array(movieIs[p][f].imarr(pol='I'))*TBfactor, cmap=cmaps[f], interpolation=interp, vmin=vmin, vmax=vmax, extent=lims)
            
            ax[i].set_title(titles[p], fontsize=18)
            ax[i].set_xticks([]), ax[i].set_yticks([])
            
            if p in polmovies.keys():
                self = polmovies[p][f]
                amp = np.sqrt(self.qvec**2 + self.uvec**2)
                scal=np.max(amp)*0.5

                vx = (-np.sin(np.angle(self.qvec+1j*self.uvec)/2)*amp/scal).reshape(self.ydim, self.xdim)
                vy = ( np.cos(np.angle(self.qvec+1j*self.uvec)/2)*amp/scal).reshape(self.ydim, self.xdim)

                # tick color will be proportional to mfrac
                mfrac=(amp/self.ivec).reshape(self.xdim, self.ydim)
                    
                pcut = 0.1
                mcut = 0.
                skip = 10
                imarr = self.imvec.reshape(self.ydim, self.xdim)
                Imax=max(self.imvec)
                mfrac = np.ma.masked_where(imarr < pcut * Imax, mfrac) 

                #new version with sharper cuts
                mfrac_map=(np.sqrt(self.qvec**2+self.uvec**2)).reshape(self.xdim, self.ydim)
                QUmax=max(np.sqrt(self.qvec**2+self.uvec**2))
                pcut=0.1
                mfrac_m = np.ma.masked_where(mfrac_map < pcut * QUmax , mfrac)
                pcut=0.1
                mfrac_m = np.ma.masked_where(imarr < pcut * Imax, mfrac_m)
                ######

                pixel=self.psize/eh.RADPERUAS #uas
                FOV=pixel*self.xdim

                # generate 2 2d grids for the x & y bounds
                y, x = np.mgrid[slice(-FOV/2, FOV/2, pixel),
                                slice(-FOV/2, FOV/2, pixel)]

                x = np.ma.masked_where(imarr < pcut * Imax, x) 
                y = np.ma.masked_where(imarr < pcut * Imax, y) 
                vx = np.ma.masked_where(imarr < pcut * Imax, vx) 
                vy = np.ma.masked_where(imarr < pcut * Imax, vy) 
                
  
                cnorm=Normalize(vmin=0.0, vmax=0.5)
                tickplot = ax[i].quiver(-x[::skip, ::skip],-y[::skip, ::skip],vx[::skip, ::skip],vy[::skip, ::skip],
                               mfrac_m[::skip,::skip],
                               headlength=0,
                               headwidth = 1,
                               pivot='mid',
                               width=0.01,
                               cmap='rainbow',
                               norm=cnorm,
                               scale=16)
        if f==0:
            ax1 = fig.add_axes([linear_interpolation(num_subplots, 2, 0.82, 7, 0.92), linear_interpolation(num_subplots, 2, 0.025, 7, 0.1), linear_interpolation(num_subplots, 2, 0.035, 7, 0.01), linear_interpolation(num_subplots, 2, 0.765, 7, 0.6)] , anchor = 'E') 
            cbar = fig.colorbar(tickplot, cmap='rainbow', cax=ax1, pad=0.14,fraction=0.038, orientation="vertical", ticklocation='right') 
            cbar.set_label('$|m|$') 
        
        plt.suptitle(f"{u_times[f]:.2f} UT", y=0.95, fontsize=22)

        return fig
    
    def update(f):
        return plot_frame(f)

    ani = animation.FuncAnimation(fig, update, frames=len(u_times), interval=1e3/fps)
    wri = animation.writers['ffmpeg'](fps=fps, bitrate=1e6)

    # Save gif
    ani.save(f'{outpath}.gif', writer=wri, dpi=100)

writegif(imlistIs, titles, paths, outpath=outpath, fov=fov, times=u_times, cmaps=cmapsl)
