##############################################################################################
# Author: Rohan Dahale, Date: 23 July 2024, Version=v1.0
##############################################################################################
import os
import glob
import ehtim as eh
import evaluation as ev

"""
# `subdir`:

Full path to the directory that contains: 
1) Truth .hdf5             | <model>_<band>_truth.uvfits (e.g. mring+hsCCW_LO_truth.hdf5)
2) Unprocessed .uvfits     | <model>_<band>.uvfits (e.g. mring+hsCCW_LO.uvfits)
3) Reconstructions .uvfits | <model>_<band>_<pipeline>.uvfits (e.g. mring+hsCCW_LO+HI_resolve.hdf5)

<model>    : crescent, disk, edisk, point, double, ring, 
             mring+hsCCW, mring+hsCW, xmasring, mring+not-center-hs, mring+4static-hs,
             SGRA, GRMHD
         
<band>     : LO, HI, LO+HI
<pipeline> : kine, resolve, ehtim, doghit, ngmem
         
         
# `resultsdir`
Full path to the directory that will contain all the results.
"""

# Submission Directory
subdir='/mnt/disks/shared/eht/sgra_dynamics_april11/mexico/submissions_test/'
# Noise and Scattering in data used for reconstruction
noise ='thermal+phasecorruptions'
scat = 'none'   # Options: sct, dsct, none
# Results Directory
resultsdir='/mnt/disks/shared/eht/sgra_dynamics_april11/mexico/results_test/'
        

eval_chisq            = True  # Chi-squares: I, Q, U, V | cphase, logcamp, amp
eval_closure_phases   = True  # Fits to closures triangles: I, Q, U, V
                              # [('AZ', 'LM', 'SM'), ('AA', 'AZ', 'SM'), ('AA', 'LM', 'SM')]
eval_amplitudes       = True      # Fits to amplitudes: I, Q, U, V
                              # [('AZ', 'LM'), ('AA', 'AZ'), ('LM', 'SM')]                          
plot_gifs             = True  # Plot Stokes I, Stokes P, Stokes V Gif: Total, Dynamic, Static
eval_nxcorr           = True  # NXCORR: Total, Static, Dynamic, NXCORR Thresholds
plot_mbreve           = True  # Plot mbreve
plot_vis_var          = True  # Plot visibility variance of truth and resconstructions
eval_rex              = True  # Ring characterization with REx in total intensity and polarization
eval_VIDA_pol         = True  # Ring characterization with VIDA in polarization
eval_VIDA             = True  # VIDA templates fits : total and dynamic component
eval_pattern_speed    = True  # Pattern speed for ring models

# Physical CPU cores to be used
cores = 100

ev.evaluation(subdir=subdir, noise=noise, scat=scat, resultsdir=resultsdir, eval_chisq=eval_chisq, 
         eval_closure_phases=eval_closure_phases, eval_amplitudes=eval_amplitudes, 
         plot_gifs=plot_gifs, eval_nxcorr=eval_nxcorr, plot_mbreve=plot_mbreve, 
         plot_vis_var=plot_vis_var, eval_rex=eval_rex, eval_VIDA_pol=eval_VIDA_pol, 
         eval_VIDA=eval_VIDA, eval_pattern_speed=eval_pattern_speed, cores=cores)