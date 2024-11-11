##############################################################################################
# Author: Rohan Dahale, Date: 09 November 2024, Version=v1.2
##############################################################################################
import os
import glob
import ehtim as eh
import evaluation as ev

"""
# `subdir`:

Full path to the directory that contains: 
1) Truth .hdf5             | <model>_<band>_<scattering>_truth.uvfits (e.g. mring+hsCCW_LO_truth.hdf5)
2) Data .uvfits     | <model>_<band>_<scattering>.uvfits (e.g. mring+hsCCW_LO_onsky.uvfits)
3) Reconstructions .hdf5 | <model>_<band>_<scattering>_<pipeline>.hdf5 (e.g. mring+hsCCW_LO+H_onsky_resolve.hdf5)

<model>      : crescent, disk, edisk, point, double, ring, 
               mring+hsCCW, mring+hsCW, mring+hs-incoh, mring+hs-cross, mring+hs-not-center, mring+hs-pol,
               sgra, grmhd
         
<band>       : LO, HI, LO+HI
<scattering> : onsky, deblur, dsct, none
<pipeline>   : kine, resolve, ehtim, doghit, ngmem
         
         
# `resultsdir`
Full path to the directory that will contain all the results.
"""

# Submission Directory
subdir='/mnt/disks/shared/eht/sgra_dynamics_april11/DAR_09Nov2024/submissions/'
# Results Directory
resultsdir='/mnt/disks/shared/eht/sgra_dynamics_april11/DAR_09Nov2024/results_test/'
        

eval_chisq            = False  # Chi-squares: I, Q, U, V | cphase, logcamp, amp
eval_closure_phases   = False  # Fits to closures triangles: I, Q, U, V
                              # [('AZ', 'LM', 'SM'), ('AA', 'AZ', 'SM'), ('AA', 'LM', 'SM')]
eval_amplitudes       = False      # Fits to amplitudes: I, Q, U, V
                              # [('AZ', 'LM'), ('AA', 'AZ'), ('LM', 'SM')]                          
plot_gifs             = True  # Plot Stokes I, Stokes P, Stokes V Gif: Total, Dynamic, Static
eval_nxcorr           = False   # NXCORR: Total, Static, Dynamic, NXCORR Thresholds
plot_mbreve           = False  # Plot mbreve
plot_vis_var          = False  # Plot visibility variance of truth and resconstructions
eval_rex              = False  # Ring characterization with REx in total intensity and polarization
eval_VIDA_pol         = False  # Ring characterization with VIDA in polarization
eval_VIDA             = False  # VIDA templates fits : total and dynamic component
eval_pattern_speed    = False  # Pattern speed for ring models

# Only when running for the first time
#setupdir=os.getcwd()+'/src'
#os.system(f'julia {setupdir}/setup.jl')

ev.evaluation(subdir=subdir, resultsdir=resultsdir, eval_chisq=eval_chisq, 
         eval_closure_phases=eval_closure_phases, eval_amplitudes=eval_amplitudes, 
         plot_gifs=plot_gifs, eval_nxcorr=eval_nxcorr, plot_mbreve=plot_mbreve, 
         plot_vis_var=plot_vis_var, eval_rex=eval_rex, eval_VIDA_pol=eval_VIDA_pol, 
         eval_VIDA=eval_VIDA, eval_pattern_speed=eval_pattern_speed)