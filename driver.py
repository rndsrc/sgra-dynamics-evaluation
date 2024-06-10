##############################################################################################
# Author: Rohan Dahale, Date: 14 May 2024, Version=v0.10
##############################################################################################

import os
import glob
import ehtim as eh


# Submission Directory
subdir='/mnt/disks/shared/eht/sgra_dynamics_april11/mexico/submissions/'

# Results Directory
resultsdir='/mnt/disks/shared/eht/sgra_dynamics_april11/mexico/results/'

##############################################################################################
# Directory of the results
##############################################################################################
if not os.path.exists(f'{resultsdir}'):
    os.makedirs(f'{resultsdir}')


# Pipeline and Colors
colors = {   
            'truth'      : 'black',
            'kine'       : 'red',
            'ehtim'      : 'forestgreen',
            'doghit'     : 'darkviolet',
            'ngmem'      : 'crimson',
            'resolve'    : 'hotpink'
        }

# Physical CPU cores to be used
cores = 100

# Dictionary of vida templates available
modelsvida={
        'crescent'  : 'mring_0_1', 
        'ring'      : 'mring_0_0', 
        'disk'      : 'disk_1', 
        'edisk'     : 'stretchdisk_1',
        'double'    : 'gauss_2', 
        'point'     : 'gauss_2',
        }

#vida_modelname = 'crescent', 'ring', 'disk', 'edisk', 'double', 'point', 'sgra'
#modeltype      = 'ring', 'non-ring' (For REx)

noise ='thermal + phase corruptions '

fulldatalist=sorted(glob.glob(subdir+'*.uvfits'))

for d in fulldatalist:
    obs = eh.obsdata.load_uvfits(d)
    if obs.tstart<10.85 or obs.tstop>14.05:
        obs = obs.flag_UT_range(UT_start_hour=10.85, UT_stop_hour=14.05, output='flagged')
        obs.tstart, obs.tstop = obs.data['time'][0], obs.data['time'][-1]
        obs.save_uvfits(d)
   
datalist=[]
for d in fulldatalist:
    if d.find('HI')==-1:
        datalist.append(d)

movielist = sorted(glob.glob(subdir+'*.hdf5'))
resolve=[]
kine=[]
ehtim=[]
doghit=[]
ngmem=[]
truth=[]

for m in movielist:
    if m.find('resolve')!=-1:
        resolve.append(m)
    elif m.find('kine')!=-1:
        kine.append(m)
    elif m.find('ngmem')!=-1:
        ngmem.append(m)
    elif m.find('doghit')!=-1:
        doghit.append(m)
    elif m.find('ehtim')!=-1:
        ehtim.append(m)
    elif m.find('truth')!=-1 and m.find('HI')==-1:
        truth.append(m)
        
movies={
        'truth'  : truth,
        'kine'   : kine,
        'resolve': resolve,
        'doghit' : doghit,
        'ngmem'  : ngmem,
        'ehtim'  : ehtim
}

for d in datalist:
    dataf = os.path.basename(d)
    datap = subdir + dataf
    print(dataf)
    
    if dataf.find('ring')!=-1 and dataf.find('mring')==-1 and dataf.find('xmasring')==-1:
        vida_modelname = 'ring'
        modeltype      = 'ring' 
    elif dataf.find('disk')!=-1 and dataf.find('edisk')== -1:
        vida_modelname = 'disk'
        modeltype      = 'non-ring' 
    elif dataf.find('edisk')!=-1:
        vida_modelname = 'edisk'
        modeltype      = 'non-ring' 
    elif dataf.find('double')!=-1:
        vida_modelname = 'double'
        modeltype      = 'non-ring' 
    elif dataf.find('point')!=-1:
        vida_modelname = 'point'
        modeltype      = 'non-ring' 
    else:
        vida_modelname = 'crescent'
        modeltype      = 'ring'
        
    modelname      = dataf[:-7]
    template       = modelsvida[vida_modelname] 
    scat = 'none'   # Options: sct, dsct, none
    
    if dataf.find('sgra') != -1 or dataf.find('SGRA') != -1 or dataf.find('hops') != -1:
        modelname='sgra'
        scat = 'dsct'
        
    mvsort = dict()
    for p in movies.keys():
        movie_types = movies[p]
        mvsort[p] = dict()
        for movie in movie_types:
            movief = os.path.basename(movie)
            model = movief.split('_')[0]
            moviep = subdir + movief
            if movief.find(model)!=-1:
                mvsort[p][model] = moviep   
                    
    model = dataf.split('_')[0]
    
    print('')
    print(f'Model: {model}')
    print(f'Type:  {modeltype}')
    print(f'VIDA Template: {template}')
    print(f'Results Directory: {resultsdir}')
    print(f'Cores: {cores}')
    print('')
    print(f'Data:  {datap}')
    print(f'Noise:  {noise}')
    print(f'Scattering Type:  {scat}')
    
    if modelname!='sgra':
        try:
            truthmv = mvsort['truth'][model]
            print(f'Truth: {truthmv}')
        except:
            truthmv='none'
            print(f'Truth: {truthmv}')
    
    for p in movies.keys():
        if p=='kine':
            try:
                kinemv = mvsort['kine'][model]
                print(f'{p}: {kinemv}')
            except:
                kinemv='none'
                print(f'{p}: {kinemv}')
        if p=='resolve':
            try:
                resmv = mvsort['resolve'][model]
                print(f'{p}: {resmv}')
            except:
                resmv='none'
                print(f'{p}: {resmv}')
        if p=='ehtim':
            try:
                ehtmv = mvsort['ehtim'][model]
                print(f'{p}: {ehtmv}')
            except:
                ehtmv='none'
                print(f'{p}: {ehtmv}')
        if p=='ngmem':
            try:
                ngmv = mvsort['ngmem'][model]
                print(f'{p}: {ngmv}')
            except:
                ngmv='none'
                print(f'{p}: {ngmv}')
        if p=='doghit':
            try:
                dogmv = mvsort['doghit'][model]
                print(f'{p}: {dogmv}')
            except:
                dogmv='none'
                print(f'{p}: {dogmv}')
                
    ##############################################################################################
    # Chi-squares, closure triangles, ampltitudes
    ##############################################################################################
    paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv}'
    data=datap
    
    pollist=['I', 'Q', 'U', 'V']
    for pol in pollist:
        #########################
        #CHISQ
        #########################
        outpath=f'{resultsdir}/{model}_chisq_{pol}'
        if not os.path.exists(outpath+'.png'):
            os.system(f'python ./src/chisq.py -d {data} {paths} -o {outpath} --pol {pol} --scat {scat}')
        
        #########################
        # CPHASE
        #########################
        outpath_tri=f'{resultsdir}/{model}_triangle_{pol}'
        if not os.path.exists(outpath_tri+'.png'):
            os.system(f'python ./src/triangles.py -d {data} {paths} -o {outpath_tri} --pol {pol} --scat {scat}')
        
        #########################
        # AMP
        #########################
        outpath_amp=f'{resultsdir}/{model}_amplitude_{pol}'
        if not os.path.exists(outpath_amp+'.png'):
            os.system(f'python ./src/amplitudes.py -d {data} {paths} -o {outpath_amp} --pol {pol} --scat {scat}')

    ##############################################################################################
    # NXCORR
    ##############################################################################################
    if modelname!='sgra':
        paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv}'
        outpath =f'{resultsdir}/{model}_nxcorr'
        if not os.path.exists(outpath+'.png'):
            os.system(f'python ./src/nxcorr.py --data {data} {paths} -o {outpath} --scat {scat}')
            
        outpath =f'{resultsdir}/{model}_nxcorr_static'
        if not os.path.exists(outpath+'.png'):
            os.system(f'python ./src/nxcorr_static.py --data {data} {paths} -o {outpath} --scat {scat}')
            
        outpath =f'{resultsdir}/{model}_nxcorr_dynamic'
        if not os.path.exists(outpath+'.png'):
            os.system(f'python ./src/nxcorr_dynamic.py --data {data} {paths} -o {outpath} --scat {scat}')
    
    ##############################################################################################
    # MBREVE
    ##############################################################################################
    paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv}'
    outpath_mbreve=f'{resultsdir}/{model}_mbreve.png'
    if not os.path.exists(outpath_mbreve):
        os.system(f'python ./src/mbreve.py -d {data} {paths} -o {outpath_mbreve} --scat {scat}')
    
    ##############################################################################################      
    # Stokes I GIF
    ##############################################################################################
    outpath =f'{resultsdir}/{model}_gif'
    if not os.path.exists(outpath+'.gif'):
        if modelname!='sgra':
            paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv}'
            os.system(f'python ./src/gif.py --data {data} {paths} -o {outpath} --scat {scat}')
        else:
            paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv}'
            os.system(f'python ./src/gif.py --data {data} {paths} -o {outpath} --scat {scat}')
            
    ##############################################################################################      
    # Stokes P GIF
    ##############################################################################################
    outpath =f'{resultsdir}/{model}_gif_lp'
    if not os.path.exists(outpath+'.gif'):
        if modelname!='sgra':
            paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv}'
            os.system(f'python ./src/gif_lp.py --data {data} {paths} -o {outpath} --scat {scat}')
        else:
            paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv}'
            os.system(f'python ./src/gif_lp.py --data {data} {paths} -o {outpath} --scat {scat}')
    
    ##############################################################################################      
    # Stokes V GIF
    ##############################################################################################
    outpath =f'{resultsdir}/{model}_gif_cp'
    if not os.path.exists(outpath+'.gif'):
        if modelname!='sgra':
            paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv}'
            os.system(f'python ./src/gif_cp.py --data {data} {paths} -o {outpath} --scat {scat}')
        else:
            paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv}'
            os.system(f'python ./src/gif_cp.py --data {data} {paths} -o {outpath} --scat {scat}')
            
    ##############################################################################################
    # Pol net, avg 
    ##############################################################################################
    outpath =f'{resultsdir}/{model}_pol'
    if not os.path.exists(outpath+'.png'):
        if modelname!='sgra':
            paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv}'
            os.system(f'python ./src/pol.py --data {data} {paths} -o {outpath} --scat {scat}')
        else:
            paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv}'
            os.system(f'python ./src/pol.py --data {data} {paths} -o {outpath} --scat {scat}')
    
    ##############################################################################################
    # REx, VIDA ring characterization
    ##############################################################################################  
    if modeltype =='ring':
        outpath =f'{resultsdir}/{model}_rex'
        if not os.path.exists(outpath+'.png') and not os.path.exists(outpath+'_pol.png'):
            if modelname!='sgra':
                paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv}'
                os.system(f'python ./src/rex.py --data {data} {paths} -o {outpath}')
            else:
                paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv}'
                os.system(f'python ./src/rex.py --data {data} {paths} -o {outpath}')

        outpath=f'{resultsdir}/{model}_vida_pol.png'
        if not os.path.exists(outpath):
            if modelname!='sgra':
                paths=f'--truthmv {truthmv} --kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv}'
                os.system(f'python ./src/vida_pol.py --data {data} {paths} -o {outpath} -c {cores} --scat {scat}')
            else:
                paths=f'--kinemv {kinemv} --dogmv {dogmv} --ngmv {ngmv} --resmv {resmv} --ehtmv {ehtmv}'
                os.system(f'python ./src/vida_pol.py --data {data} {paths} -o {outpath} -c {cores} --scat {scat}')

    ##############################################################################################
    # VIDA
    ##############################################################################################  
    plist=[]
    nlist=[]
    
    if modelname!='sgra':
        if truthmv!='none':
            plist.append(truthmv)
            nlist.append('truth')
            truthcsv=f'{resultsdir}/{model}_vida_truth.csv'
        else:
            truthcsv='none'
            
    if kinemv!='none':
        plist.append(kinemv)
        nlist.append('kine')
        kinecsv=f'{resultsdir}/{model}_vida_kine.csv'
    else:
        kinecsv='none'
    if resmv!='none':
        plist.append(resmv)
        nlist.append('resolve')
        rescsv=f'{resultsdir}/{model}_vida_resolve.csv'
    else:
        rescsv='none'
    if ehtmv!='none':
        plist.append(ehtmv)
        nlist.append('ehtim')
        ehtcsv=f'{resultsdir}/{model}_vida_ehtim.csv'
    else:
        ehtcsv='none'
    if dogmv!='none':
        plist.append(dogmv)
        nlist.append('doghit')
        dogcsv=f'{resultsdir}/{model}_vida_doghit.csv'
    else:
        dogcsv='none'
    if ngmv!='none':
        plist.append(ngmv)
        nlist.append('ngmem')
        ngcsv=f'{resultsdir}/{model}_vida_ngmem.csv'
    else:
        ngcsv='none'
    
    for i in range(len(plist)):
        input  = plist[i]
        output = f'{resultsdir}/{model}_vida_{nlist[i]}.csv'
        if not os.path.exists(output):
            if nlist[i]=='truth':
                os.system(f'julia -p {cores} ./src/movie_extractor_parallel.jl --input {input} --output {output} --template {template} --stride {cores} --blur 15.0')
            else:
                os.system(f'julia -p {cores} ./src/movie_extractor_parallel.jl --input {input} --output {output} --template {template} --stride {cores}')

    outpath =f'{resultsdir}/{model}_vida'
    if not os.path.exists(outpath):
        if modelname!='sgra':
            paths=f'--truthcsv {truthcsv} --kinecsv {kinecsv} --rescsv {rescsv} --ehtcsv {ehtcsv} --dogcsv {dogcsv} --ngcsv {ngcsv}'
            os.system(f'python ./src/vida.py --model {vida_modelname} {paths} -o {outpath}')
        else:
            paths=f'--kinecsv {kinecsv} --rescsv {rescsv} --ehtcsv {ehtcsv} --dogcsv {dogcsv} --ngcsv {ngcsv}'
            os.system(f'python ./src/vida.py --model {vida_modelname} {paths} -o {outpath}')

    ##############################################################################################
    # Interpolated Movie, Averaged Movie, VIDA Ring, Cylinder
    ##############################################################################################
    if modeltype =='ring':
        if not os.path.exists(f'{resultsdir}/patternspeed'):
                os.makedirs(f'{resultsdir}/patternspeed')
                
        for i in range(len(plist)):
            
            # Interpolated Movies
            input= plist[i]
            output=f'{resultsdir}/patternspeed/{os.path.basename(plist[i])}'
            os.system(f'python ./src/hdf5_standardize.py -i {input} -o {output}')
    
            #Average Movies
            input=f'{resultsdir}/patternspeed/{os.path.basename(plist[i])}'
            fits=os.path.basename(plist[i])[:-5]+'.fits'
            output=f'{resultsdir}/patternspeed/{fits}'
            if nlist[i]=='truth':
                os.system(f'python ./src/avg_frame.py -i {input} -o {output} --truth')
            else:
                os.system(f'python ./src/avg_frame.py -i {input} -o {output}')
    
            # VIDA Ring
            fits=os.path.basename(plist[i])[:-5]+'.fits'
            path=f'{resultsdir}/patternspeed/{fits}'
            outpath = path[:-5]+'.csv'
            if not os.path.exists(outpath):    
                os.system(f'julia ./src/ring_extractor.jl --in {path} --out {outpath}')
                print(f'{os.path.basename(outpath)} created!')
                
            # Cylinder
            ipathmov=f'{resultsdir}/patternspeed/{os.path.basename(plist[i])}'
            ringpath = ipathmov[:-5]+'.csv'
            outpath  = ipathmov[:-5]
            os.system(f'python ./src/cylinder.py {ipathmov} {ringpath} {outpath}')
##############################################################################################