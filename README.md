# Evaluation Script for EHT Movies

- Clone this repository
- Create a `python 3.9` conda environment with the following dependencies:
    - `pip instal ehtim==1.2.4 numpy==1.21.1 pandas==1.5.3 scipy==1.10.1 autograd networkx scikit-image tqdm jupyter`
    - `pynfft`
    - `ffmpeg`
    - [ehtplot](https://github.com/liamedeiros/ehtplot)
- Install Julia 1.10.4 with [juliaup](https://github.com/JuliaLang/juliaup). `juliaup add 1.10.4`, `juliaup default 1.10.4`
- You just need to modify `driver.py` and run it.

## <span style="color:red"> NOTE: When running `driver.py` for the first time, uncomment the lines for `setup.jl` </span>


# How to use the driver?

### 1. Define submission folder: `subdir`
Full path to the directory that contains: 
1) Truth .hdf5             | `<model>_<band>_<scattering>_truth.uvfits` (e.g. mring+hsCCW_LO_onsky_truth.hdf5)
2) Data .uvfits     | `<model>_<band>_<scattering>.uvfits` (e.g. mring+hsCCW_LO_onsky.uvfits)
3) Reconstructions .hdf5 | `<model>_<band>_<scattering>_<pipeline>.hdf`5` (e.g. mring+hsCCW_LO+HI_onsky_resolve.hdf5)

```
<model>      : crescent, disk, edisk, point, double, ring, 
               mring+hsCCW, mring+hsCW, mring+hs-incoh, mring+hs-cross, mring+hs-not-center, mring+hs-pol,
               sgra, grmhd1, grmhd2, grmhd3, grmhd2+hs
         
<band>       : LO, HI, LO+HI
<scattering> : onsky, deblur, dsct, none
<pipeline>   : kine, resolve, ehtim, doghit, ngmem
```

---         
### 2. Define results folder: `resultsdir`
Full path to the directory that will contain all the results.

---
### 3. Select the things to evaluate

```
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
```

### 4. Define the number of cores used by VIDA.jl

``` cores = 32 ```