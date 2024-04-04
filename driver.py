##############################################################################################
# Author: Rohan Dahale, Date: 25 March 2024, Version=v0.9
##############################################################################################

import os

#Base Directory of the Project
basedir='/mnt/disks/shared/eht/sgra_dynamics_april11'

# Pipeline and Colors
colors = {   
            'truth'      : 'white',
            'kine'       : 'darkorange',
            'starwarps'  : 'xkcd:azure',
            'ehtim'      : 'forestgreen',
            'doghit'     : 'darkviolet',
            'ngmem'      : 'crimson',
            'resolve'    : 'hotpink'
        }


# Results Directory
resultsdir='results_VL_1'
# Submission Directory
subdir='submission_VL_1'

models={
        'crescent'  : 'ring', 
        'ring'      : 'ring', 
        'disk'      : 'non-ring', 
        'edisk'     : 'non-ring',
        'double'    : 'non-ring', 
        'point'     : 'non-ring'
        }


modelsvida={
        'crescent'  : 'mring_0_1', 
        'ring'      : 'mring_0_0', 
        'disk'      : 'disk_1', 
        'edisk'     : 'stretchdisk_1',
        'double'    : 'gauss_2', 
        'point'     : 'gauss_2'
        }


epoch='3601'
band='LO'
cband='HI+LO'
noise='thermal+phasegains'
scat = 'none'   # Options: sct, dsct, none
cores = 100

##############################################################################################
# Directory of the results
##############################################################################################

if not os.path.exists(f'{basedir}/evaluation/{resultsdir}'):
    os.makedirs(f'{basedir}/evaluation/{resultsdir}')
    if not os.path.exists(f'{basedir}/evaluation/{resultsdir}/interpolated_movies'):
        os.makedirs(f'{basedir}/evaluation/{resultsdir}/interpolated_movies')
        for pipe in colors.keys():
            if not os.path.exists(f'{basedir}/evaluation/{resultsdir}/interpolated_movies/{pipe}'):
                os.makedirs(f'{basedir}/evaluation/{resultsdir}/interpolated_movies/{pipe}')
                
    if not os.path.exists(f'{basedir}/evaluation/{resultsdir}/averaged_movies'):
        os.makedirs(f'{basedir}/evaluation/{resultsdir}/averaged_movies')
        for pipe in colors.keys():
            if not os.path.exists(f'{basedir}/evaluation/{resultsdir}/averaged_movies/{pipe}'):
                os.makedirs(f'{basedir}/evaluation/{resultsdir}/averaged_movies/{pipe}')
                
    if not os.path.exists(f'{basedir}/evaluation/{resultsdir}/VIDA'):
        os.makedirs(f'{basedir}/evaluation/{resultsdir}/VIDA')
        for pipe in colors.keys():
            if not os.path.exists(f'{basedir}/evaluation/{resultsdir}/VIDA/{pipe}'):
                os.makedirs(f'{basedir}/evaluation/{resultsdir}/VIDA/{pipe}')
                
    if not os.path.exists(f'{basedir}/evaluation/{resultsdir}/plots'):
        os.makedirs(f'{basedir}/evaluation/{resultsdir}/plots')

##############################################################################################
# Interpolated Movies, Averaged Movies
##############################################################################################

modelname={}
for m in models.keys():
    modelname[m]={}
    for pipe in colors.keys():
        if pipe=='resolve':
            model=f'{m}_{epoch}_{cband}'
            modelname[m][pipe]=model
        else:
            model=f'{m}_{epoch}_{band}'
            modelname[m][pipe]=model
            
        # Interpolated Movies
        if pipe!='truth':
            indir=f'{basedir}/{subdir}/{pipe}/{model}_{noise}/'
            outdir=f'{basedir}/evaluation/{resultsdir}/interpolated_movies/{pipe}/{model}_{noise}/'
        else:
            indir=f'{basedir}/{subdir}/{pipe}/{model}'
            outdir=f'{basedir}/evaluation/{resultsdir}/interpolated_movies/{pipe}/{model}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        #os.system(f'python {basedir}/evaluation/scripts/sgra-dynamics-evaluation/src/hdf5_standardize.py -i {indir} -o {outdir}')
        
        #Average Movies
        if pipe!='truth':
            outdir=f'{basedir}/evaluation/{resultsdir}/averaged_movies/{pipe}/{model}_{noise}/'
        else:
            outdir=f'{basedir}/evaluation/{resultsdir}/averaged_movies/{pipe}/{model}'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        #os.system(f'python {basedir}/evaluation/scripts/sgra-dynamics-evaluation/src/avg_frame.py -i {indir} -o {outdir}')

##############################################################################################
# Chi-squares, closure triangles, ampltitudes, nxcorr, gif, pol net avg, REx, VIDA
##############################################################################################
for m in models: 
    if scat!='none':       
        pathmov  = f'{basedir}/{subdir}/kine/{modelname[m]["kine"]}_{noise}/{modelname[m]["kine"]}_1_{scat}.hdf5'
        pathmov2 = f'{basedir}/{subdir}/starwarps/{modelname[m]["starwarps"]}_{noise}/{modelname[m]["starwarps"]}_1_{scat}.hdf5'
        pathmov3 = f'{basedir}/{subdir}/ehtim/{modelname[m]["ehtim"]}_{noise}/{modelname[m]["ehtim"]}_1_{scat}.hdf5'
        pathmov4 = f'{basedir}/{subdir}/doghit/{modelname[m]["doghit"]}_{noise}/{modelname[m]["doghit"]}_1_{scat}.hdf5'
        pathmov5 = f'{basedir}/{subdir}/ngmem/{modelname[m]["ngmem"]}_{noise}/{modelname[m]["ngmem"]}_1_{scat}.hdf5'
        pathmov6 = f'{basedir}/{subdir}/resolve/{modelname[m]["resolve"]}_{noise}/{modelname[m]["resolve"]}_1_{scat}.hdf5'
    else:
        pathmov  = f'{basedir}/{subdir}/kine/{modelname[m]["kine"]}_{noise}/{modelname[m]["kine"]}_1.hdf5'
        pathmov2 = f'{basedir}/{subdir}/starwarps/{modelname[m]["starwarps"]}_{noise}/{modelname[m]["starwarps"]}_1.hdf5'
        pathmov3 = f'{basedir}/{subdir}/ehtim/{modelname[m]["ehtim"]}_{noise}/{modelname[m]["ehtim"]}_1.hdf5'
        pathmov4 = f'{basedir}/{subdir}/doghit/{modelname[m]["doghit"]}_{noise}/{modelname[m]["doghit"]}_1.hdf5'
        pathmov5 = f'{basedir}/{subdir}/ngmem/{modelname[m]["ngmem"]}_{noise}/{modelname[m]["ngmem"]}_1.hdf5'
        pathmov6 = f'{basedir}/{subdir}/resolve/{modelname[m]["resolve"]}_{noise}/{modelname[m]["resolve"]}_1.hdf5'
    
   #paths=f'--kinemv {pathmov} --starmv {pathmov2} --ehtmv {pathmov3} --dogmv {pathmov4} --ngmv {pathmov5} --resmv {pathmov6}'
    paths=f'--kinemv {pathmov} --dogmv {pathmov4} --ngmv {pathmov5} --resmv {pathmov6}'
    
    data=f'{basedir}/{subdir}/data/{m}_{epoch}_{band}.uvfits'
    
    pollist=['I', 'Q', 'U', 'V']
    for pol in pollist:
        #CHISQ
        outpath=f'{basedir}/evaluation/{resultsdir}/plots/chisq_{pol}_{modelname[m]["kine"]}'
        if not os.path.exists(outpath+'.png'):
            os.system(f'python {basedir}/evaluation/scripts/sgra-dynamics-evaluation/src/chisq.py -d {data} {paths} -o {outpath} --pol {pol} --scat {scat}')
        
        # CPHASE
        outpath_tri=f'{basedir}/evaluation/{resultsdir}/plots/triangle_{pol}_{modelname[m]["kine"]}'
        if not os.path.exists(outpath_tri+'.png'):
            os.system(f'python {basedir}/evaluation/scripts/sgra-dynamics-evaluation/src/triangles.py -d {data} {paths} -o {outpath_tri} --pol {pol} --scat {scat}')
        
        # AMP
        outpath_amp=f'{basedir}/evaluation/{resultsdir}/plots/amplitude_{pol}_{modelname[m]["kine"]}'
        if not os.path.exists(outpath_amp+'.png'):
            os.system(f'python {basedir}/evaluation/scripts/sgra-dynamics-evaluation/src/amplitudes.py -d {data} {paths} -o {outpath_amp} --pol {pol} --scat {scat}')

    # NXCORR
    if scat!='none':
        pathmovt  = f'{basedir}/{subdir}/truth/{modelname[m]["truth"]}/{modelname[m]["truth"]}_{scat}.hdf5'
    else:
        pathmovt  = f'{basedir}/{subdir}/truth/{modelname[m]["truth"]}/{modelname[m]["truth"]}.hdf5'

    #paths=f'--truthmv {pathmovt} --kinemv {pathmov} --starmv {pathmov2} --ehtmv {pathmov3} --dogmv {pathmov4} --ngmv {pathmov5} --resmv {pathmov6}'
    paths=f'--truthmv {pathmovt} --kinemv {pathmov} --dogmv {pathmov4} --ngmv {pathmov5} --resmv {pathmov6}'
    
    outpath =f'{basedir}/evaluation/{resultsdir}/plots/nxcorr_{modelname[m]["kine"]}'
    if not os.path.exists(outpath+'.png'):
        os.system(f'python {basedir}/evaluation/scripts/sgra-dynamics-evaluation/src/nxcorr.py --data {data} {paths} -o {outpath} --scat {scat}')
          
    # Stokes I GIF  
    outpath =f'{basedir}/evaluation/{resultsdir}/plots/gif_{modelname[m]["truth"]}'
    if not os.path.exists(outpath+'.gif'):
        os.system(f'python {basedir}/evaluation/scripts/sgra-dynamics-evaluation/src/gif.py --data {data} {paths} -o {outpath} --scat {scat}')
    
    # Stokes P GIF 
    outpath =f'{basedir}/evaluation/{resultsdir}/plots/gif_lp_{modelname[m]["truth"]}'
    if not os.path.exists(outpath+'.gif'):
        os.system(f'python {basedir}/evaluation/scripts/sgra-dynamics-evaluation/src/gif_lp.py --data {data} {paths} -o {outpath} --scat {scat}')
    
    # Stokes V GIF 
    outpath =f'{basedir}/evaluation/{resultsdir}/plots/gif_cp_{modelname[m]["truth"]}'
    if not os.path.exists(outpath+'.gif'):
        os.system(f'python {basedir}/evaluation/scripts/sgra-dynamics-evaluation/src/gif_cp.py --data {data} {paths} -o {outpath} --scat {scat}')
    
    # Pol net, avg 
    outpath =f'{basedir}/evaluation/{resultsdir}/plots/pol_{modelname[m]["kine"]}'
    if not os.path.exists(outpath+'.png'):
        os.system(f'python {basedir}/evaluation/scripts/sgra-dynamics-evaluation/src/pol.py --data {data} {paths} -o {outpath} --scat {scat}')
        
    # REx ring characterization
    if models[m] =='ring':
        outpath =f'{basedir}/evaluation/{resultsdir}/plots/rex_{modelname[m]["kine"]}'
        if not os.path.exists(outpath+'.png') and not os.path.exists(outpath+'_pol.png'):
            os.system(f'python {basedir}/evaluation/scripts/sgra-dynamics-evaluation/src/rex.py --data {data} {paths} -o {outpath}')
            
    # VIDA
    for pipe in colors.keys():
        if pipe!='truth':
            inputdir  = f'{basedir}/{subdir}/{pipe}/{modelname[m][pipe]}_{noise}/'
            outputdir =f'{basedir}/evaluation/{resultsdir}/VIDA/{pipe}/{modelname[m][pipe]}_{noise}/'
        else:
            inputdir  = f'{basedir}/{subdir}/{pipe}/{modelname[m][pipe]}/'
            outputdir =f'{basedir}/evaluation/{resultsdir}/VIDA/{pipe}/{modelname[m][pipe]}/'
    
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
            os.system(f'julia -p {cores} {basedir}/evaluation/scripts/sgra-dynamics-evaluation/src/movie_extractor_parallel.jl --inputdir {inputdir} --outputdir {outputdir} --template {modelsvida[m]} --stride {cores} --scat {scat}')
            
    if scat!='none':       
        truthcsv  = f'{basedir}/evaluation/{resultsdir}/VIDA/truth/{modelname[m]["truth"]}/{modelname[m]["truth"]}_{scat}_vida.csv'
        kinecsv    = f'{basedir}/evaluation/{resultsdir}/VIDA/kine/{modelname[m]["kine"]}_{noise}/{modelname[m]["kine"]}_1_{scat}_vida.csv'
        starcsv   = f'{basedir}/evaluation/{resultsdir}/VIDA/starwarps/{modelname[m]["starwarps"]}_{noise}/{modelname[m]["starwarps"]}_1_{scat}_vida.csv'
        ehtcsv    = f'{basedir}/evaluation/{resultsdir}/VIDA/ehtim/{modelname[m]["ehtim"]}_{noise}/{modelname[m]["ehtim"]}_1_{scat}_vida.csv'
        dogcsv    = f'{basedir}/evaluation/{resultsdir}/VIDA/doghit/{modelname[m]["doghit"]}_{noise}/{modelname[m]["doghit"]}_1_{scat}_vida.csv'
        ngcsv     = f'{basedir}/evaluation/{resultsdir}/VIDA/ngmem/{modelname[m]["ngmem"]}_{noise}/{modelname[m]["ngmem"]}_1_{scat}_vida.csv'
        rescsv    = f'{basedir}/evaluation/{resultsdir}/VIDA/resolve/{modelname[m]["resolve"]}_{noise}/{modelname[m]["resolve"]}_1_{scat}_vida.csv'
    else:
        truthcsv  = f'{basedir}/evaluation/{resultsdir}/VIDA/truth/{modelname[m]["truth"]}/{modelname[m]["truth"]}_vida.csv'
        kinecsv   = f'{basedir}/evaluation/{resultsdir}/VIDA/kine/{modelname[m]["kine"]}_{noise}/{modelname[m]["kine"]}_1_vida.csv'
        starcsv   = f'{basedir}/evaluation/{resultsdir}/VIDA/starwarps/{modelname[m]["starwarps"]}_{noise}/{modelname[m]["starwarps"]}_1_vida.csv'
        ehtcsv    = f'{basedir}/evaluation/{resultsdir}/VIDA/ehtim/{modelname[m]["ehtim"]}_{noise}/{modelname[m]["ehtim"]}_1_vida.csv'
        dogcsv    = f'{basedir}/evaluation/{resultsdir}/VIDA/doghit/{modelname[m]["doghit"]}_{noise}/{modelname[m]["doghit"]}_1_vida.csv'
        ngcsv     = f'{basedir}/evaluation/{resultsdir}/VIDA/ngmem/{modelname[m]["ngmem"]}_{noise}/{modelname[m]["ngmem"]}_1_vida.csv'
        rescsv    = f'{basedir}/evaluation/{resultsdir}/VIDA/resolve/{modelname[m]["resolve"]}_{noise}/{modelname[m]["resolve"]}_1_vida.csv'
       
    
    outpath =f'{basedir}/evaluation/{resultsdir}/plots/vida_{modelname[m]["kine"]}'
    paths=f'--truthcsv {truthcsv} --kinecsv {kinecsv} --dogcsv {dogcsv} --ngcsv {ngcsv} --rescsv {rescsv}'
    if not os.path.exists(outpath+'.png'):    
        os.system(f'python {basedir}/evaluation/scripts/sgra-dynamics-evaluation/src/vida.py --model {m} {paths} -o {outpath}')
        
        
    #########################################################
    # Cylinder
    #########################################################
    if models[m] =='ring':
        # Run VIDA on average movies for ring parameters to be used by cylinder
        if scat!='none':
            pathmovt = f'{basedir}/evaluation/{resultsdir}/averaged_movies/truth/{modelname[m]["truth"]}/{modelname[m]["truth"]}_{scat}.fits'       
            pathmov  = f'{basedir}/evaluation/{resultsdir}/averaged_movies/kine/{modelname[m]["kine"]}_{noise}/{modelname[m]["kine"]}_1_{scat}.fits'
            pathmov2 = f'{basedir}/evaluation/{resultsdir}/averaged_movies/starwarps/{modelname[m]["starwarps"]}_{noise}/{modelname[m]["starwarps"]}_1_{scat}.fits'
            pathmov3 = f'{basedir}/evaluation/{resultsdir}/averaged_movies/ehtim/{modelname[m]["ehtim"]}_{noise}/{modelname[m]["ehtim"]}_1_{scat}.fits'
            pathmov4 = f'{basedir}/evaluation/{resultsdir}/averaged_movies/doghit/{modelname[m]["doghit"]}_{noise}/{modelname[m]["doghit"]}_1_{scat}.fits'
            pathmov5 = f'{basedir}/evaluation/{resultsdir}/averaged_movies/ngmem/{modelname[m]["ngmem"]}_{noise}/{modelname[m]["ngmem"]}_1_{scat}.fits'
            pathmov6 = f'{basedir}/evaluation/{resultsdir}/averaged_movies/resolve/{modelname[m]["resolve"]}_{noise}/{modelname[m]["resolve"]}_1_{scat}.fits'
        else:
            pathmovt = f'{basedir}/evaluation/{resultsdir}/averaged_movies/truth/{modelname[m]["truth"]}/{modelname[m]["truth"]}.fits'
            pathmov  = f'{basedir}/evaluation/{resultsdir}/averaged_movies/kine/{modelname[m]["kine"]}_{noise}/{modelname[m]["kine"]}_1.fits'
            pathmov2 = f'{basedir}/evaluation/{resultsdir}/averaged_movies/starwarps/{modelname[m]["starwarps"]}_{noise}/{modelname[m]["starwarps"]}_1.fits'
            pathmov3 = f'{basedir}/evaluation/{resultsdir}/averaged_movies/ehtim/{modelname[m]["ehtim"]}_{noise}/{modelname[m]["ehtim"]}_1.fits'
            pathmov4 = f'{basedir}/evaluation/{resultsdir}/averaged_movies/doghit/{modelname[m]["doghit"]}_{noise}/{modelname[m]["doghit"]}_1.fits'
            pathmov5 = f'{basedir}/evaluation/{resultsdir}/averaged_movies/ngmem/{modelname[m]["ngmem"]}_{noise}/{modelname[m]["ngmem"]}_1.fits'
            pathmov6 = f'{basedir}/evaluation/{resultsdir}/averaged_movies/resolve/{modelname[m]["resolve"]}_{noise}/{modelname[m]["resolve"]}_1.fits'
            
        paths=[pathmovt, pathmov, pathmov4, pathmov5, pathmov6]
        for path in paths:
            outpath = path[:-5]+'.csv'
            if not os.path.exists(outpath):    
                os.system(f'julia {basedir}/evaluation/scripts/sgra-dynamics-evaluation/src/ring_extractor.jl --in {path} --out {outpath}')
                print(f'{os.path.basename(outpath)} created!')
         
        ######################
        # For Nick: Below this
        ######################
           
        # Run cylinder on all interpolated movies
        if scat!='none':       
            pathmov  = f'{basedir}/evalutation/{resultsdir}/interpolated_movies/kine/{modelname[m]["kine"]}_{noise}/{modelname[m]["kine"]}_1_{scat}.hdf5'
            pathmov2 = f'{basedir}/evalutation/{resultsdir}/interpolated_movies/starwarps/{modelname[m]["starwarps"]}_{noise}/{modelname[m]["starwarps"]}_1_{scat}.hdf5'
            pathmov3 = f'{basedir}/evalutation/{resultsdir}/interpolated_movies/ehtim/{modelname[m]["ehtim"]}_{noise}/{modelname[m]["ehtim"]}_1_{scat}.hdf5'
            pathmov4 = f'{basedir}/evalutation/{resultsdir}/interpolated_movies/doghit/{modelname[m]["doghit"]}_{noise}/{modelname[m]["doghit"]}_1_{scat}.hdf5'
            pathmov5 = f'{basedir}/evalutation/{resultsdir}/interpolated_movies/ngmem/{modelname[m]["ngmem"]}_{noise}/{modelname[m]["ngmem"]}_1_{scat}.hdf5'
            pathmov6 = f'{basedir}/evalutation/{resultsdir}/interpolated_movies/resolve/{modelname[m]["resolve"]}_{noise}/{modelname[m]["resolve"]}_1_{scat}.hdf5'
        else:
            pathmov  = f'{basedir}/evaluation/{resultsdir}/interpolated_movies/kine/{modelname[m]["kine"]}_{noise}/{modelname[m]["kine"]}_1.hdf5'
            pathmov2 = f'{basedir}/evaluation/{resultsdir}/interpolated_movies/starwarps/{modelname[m]["starwarps"]}_{noise}/{modelname[m]["starwarps"]}_1.hdf5'
            pathmov3 = f'{basedir}/evaluation/{resultsdir}/interpolated_movies/ehtim/{modelname[m]["ehtim"]}_{noise}/{modelname[m]["ehtim"]}_1.hdf5'
            pathmov4 = f'{basedir}/evaluation/{resultsdir}/interpolated_movies/doghit/{modelname[m]["doghit"]}_{noise}/{modelname[m]["doghit"]}_1.hdf5'
            pathmov5 = f'{basedir}/evaluation/{resultsdir}/interpolated_movies/ngmem/{modelname[m]["ngmem"]}_{noise}/{modelname[m]["ngmem"]}_1.hdf5'
            pathmov6 = f'{basedir}/evaluation/{resultsdir}/interpolated_movies/resolve/{modelname[m]["resolve"]}_{noise}/{modelname[m]["resolve"]}_1.hdf5'

        # Run cylinder on all these interpolated movie paths
        paths=[pathmovt, pathmov, pathmov4, pathmov5, pathmov6]
        
        #for path in paths:
            #os.system(f'python {basedir}/evaluation/scripts/sgra-dynamics-evaluation/src/cylinder.py')
    
          
##############################################################################################