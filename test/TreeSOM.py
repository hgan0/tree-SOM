from pathlib import Path
from tqdm.notebook import tqdm
from astropy.io import fits
from minisom import MiniSom
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from joblib import Parallel, delayed
from joblib_progress import joblib_progress

import os
import pandas as pd
import minisom 
import pickle
import random
import time


def lscale(x):
    x = np.abs(x)
    return np.log10(x / x.max())

def trim_data(data):
    new_data = data.replace(' -inf', np.nan)
    new_data = new_data.replace(' inf', np.nan)
    new_data = new_data.replace(' -nan ',np.nan)
    new_data = new_data.replace(' nan ',np.nan)
    new_data = new_data.replace(np.nan, 0)
    new_data = new_data.astype(float)
    return new_data 

def LoadImageDirectory(
    img_directory: str, 
    extension: str
):
    """
    Parameters
    ----------
    directory 
            image directory
    extension 
            image extension
    """
    imagefiles = sorted(Path(img_directory).glob("*."+extension))
    return imagefiles
    
def CreateFitsforCHIME(
    ids, 
    base_directory: str,
    revision = "rev_07",
):
    for ii, csd in tqdm(enumerate(ids)):
        csd = str(csd)
        daily_processing_path = Path(f"/project/rpp-chime/chime/chime_processed/daily/{revision}/")
    
        try:
            name = "delayspectrum_lsd_" + csd + ".h5"
            ds = containers.DelaySpectrum.from_file( daily_processing_path / csd / name)
            newimage = lscale(ds.spectrum[:].T)
            os.makedirs(base_directory+'/fits/', exist_ok=True)
            astropy.io.fits.writeto(base_directory+'/fits/'+csd+'.fits', newimage, header=None,)            
        except: 
            print("Failed to create a fit file!")

def CreateTrees(
    imagefiles,
    extension: str,
    disccofan_directory: str,
    base_directory: str,
    n_thread = 1, 
    c = 8,
    a = 12,
    lval = 100,
):
    """
    Parameters (for details see https://github.com/sgazagnes/disccofan) 
    ----------
    c (connectivity) : Type of connectivity:
    4:  4 connectivity for 2D dataset (default for 2D)
    6:  6 connectivity for 3D dataset (default for 3D)
    8:  8 connectivity for 2D dataset
           26: 26 connectivity for 3D dataset
        a (attributes) : Choose the attribute used in the tree nodes
    0: Area (default)
    1: Area of min enclosing rectangle
    2: Square of diagonal of minimal enclosing rectangle
    3: Moment of inertia
    4: (Moment of Inertia) / (area)^2
    5: Mean X position
    6: Mean Y position
    7: Mean Z position
    8 - Full inertia: area (same as attribute 0)
    9 - Full inertia: elongation
    10 - Full inertia: flatness
    11 - Full inertia: sparseness
    12 - Full inertia: Ncompactness
        lval (lambda value) : Threshold value for pruning
    """
    os.makedirs(base_directory + '/trees/', exist_ok=True) 
    disccofan_directory = disccofan_directory + '/disccofan'
    for file in tqdm(imagefiles):
        filename = file.stem
        filedir = os.path.splitext(file)[0]
        outname = base_directory + '/trees/' + filename
        command = '%s -g 1,1,1 --threads %d  -c %d  -a %d  --inprefix %s --intype %s -o treeCSV --outprefix %s -l %d'%(disccofan_directory, n_thread, c, a, filedir, extension, outname, lval)
        try: 
            os.system(command)
        except:
            print("Failed to create a tree..!")
            
def ShowImages(
    imagefiles
):
    # setting values to rows and column variables
    rows = round(np.sqrt(len(imagefiles)))
    columns = round(np.sqrt(len(imagefiles)))
    
    fig, axs = plt.subplots(rows, columns, figsize=(10, 8) )
    fig.subplots_adjust(hspace = .1, wspace=.05)
    axs = axs.ravel()
    
    for ii, img_i in enumerate(imagefiles):
        #fig.add_subplot(rows, columns, ii+1)
        axs[ii].imshow(mpimg.imread(img_i), cmap='Greys_r') #cv2.imread(img_i))
        axs[ii].axis('off')
        axs[ii].set_title(img_i.stem)
    plt.show()
    
    
def LoadTrees(
    base_directory : str,
    extension = "csv", 
    sampling_rate = 1.0,     
):
    treefiles = sorted(Path(base_directory+'/trees/').glob("*."+extension))
    data = pd.DataFrame()    
    
    for tree_file in tqdm(treefiles):
        tmp = pd.read_csv(tree_file, sep=',', header=0, low_memory=False) 
        tmp = trim_data(tmp)     
        sampled_tmp = tmp.sample(n = round(len(tmp.index)*sampling_rate) )
        data = pd.concat([data, sampled_tmp], ignore_index=True)
    print(str(len(data)) + ' nodes are loaded in total!')
    return data


def TrainSOM(
    train_set,
    base_directory,
    n_neurons,
    m_neurons,
    s_features = [4,5,6,8,9,13,14],
    max_iter = 1000,
    sigma = 1. ,
    learning_rate = .2, 
    neighborhood_function = 'gaussian',
    random_seed = 0,
    topology = 'rectangular',
    activation_distance = 'euclidean',
):
    """
    Parameters (for details see https://github.com/sgazagnes/disccofan) 
    ----------
    s_features:
    
    0 : node index 
    1 : parent
    2 : grey value
    3 : alpha
    4 : flux
    5 : volume
    6 : elongation
    7 : flatness
    8 : sparseness
    9 : non-compactness
    10 : centre of mass in x-axis 
    11 : centre of mass in y-axis 
    12 : centre of mass in z-axis 
    13 : weighted centre of mass in x-axis 
    14 : weighted centre of mass in x-axis 
    15 : weighted centre of mass in x-axis

    """
    flat_data = []
    features = np.array(list(train_set.keys()))
    print("Selected training attributes are : ")
    print(features[s_features])
    
    for i in range(0,len(features)):
        flat_data.append(train_set.iloc[:,i].to_numpy())
    X = np.array(flat_data)[s_features]
    X = np.array(X).T
    scaled_X = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0) 
   
    som = MiniSom(n_neurons, m_neurons, len(scaled_X[0]), activation_distance=activation_distance, 
          sigma=sigma, learning_rate=learning_rate, topology=topology,
          neighborhood_function=neighborhood_function, random_seed=random_seed)
    som.pca_weights_init(scaled_X)
    som.train(scaled_X, max_iter, verbose=True)  # random training    
    q_error = som.quantization_error(scaled_X)
    plt.figure(figsize=(6, 5))
    plt.pcolor(som.distance_map().T[::-1,], cmap='bone_r')  # plotting the distance map as background
    plt.xlabel('N_x', fontsize=12)
    plt.ylabel('N_y', fontsize=12)
    plt.title('Distance map', fontsize=14)
    plt.colorbar()
    plt.savefig(base_directory + '/SOM_distance_map.pdf')
    with open(base_directory+'/som.p', 'wb') as outfile:
        pickle.dump(som, outfile)
    print('Training completed!')
    
    
def GetWinningNeurons(
    base_directory,
    s_features,
    n_neurons,
    m_neurons,
    tree_extension = 'csv'
):
    tree_directory = base_directory + '/trees/' 
    file_names = sorted(Path(tree_directory).glob("*."+tree_extension))
    
    with open(base_directory+'/som.p', 'rb') as infile:
        som = pickle.load(infile)    

    for file in file_names:
        filename = file.stem
        single_data = pd.read_csv(file, sep=',', header=0, low_memory=False) 
        single_data = trim_data(single_data)
        features = np.array(list(single_data.keys()))

        n_columns = np.shape(single_data)[-1]
        flat_data = []

        for i in range(0,len(features)):
            flat_data.append(single_data.iloc[:,i].to_numpy())

        X = np.array(flat_data)[s_features]
        X = np.array(X).T
        scaled_X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  

        W = som.get_weights()
        w_x, w_y = zip(*[som.winner(d) for d in scaled_X])
        w_x = np.array(w_x)
        w_y = np.array(w_y)  

        df2 =  pd.DataFrame({'index': flat_data[0],
               'winning_Nx': w_x,
               'winning_Ny': w_y,
                            })
        os.makedirs(base_directory+'/SOMs/', exist_ok=True)  
        df2 = df2.loc[:, ~df2.columns.str.contains('^Unnamed')]
        df2 = df2.astype(int)
        df2.to_csv(base_directory+'/SOMs/'+filename+'.csv') 

def GetWinningNeurons_in_Parallel(
    base_directory,
    s_features,
    n_neurons,
    m_neurons,
    tree_extension = 'csv'
):
    tree_directory = base_directory + '/trees/' 
    file_names = sorted(Path(tree_directory).glob("*."+tree_extension))
    
    with joblib_progress("Calculating winning neurons..."):
        element_run = Parallel(n_jobs=-1)(delayed(GetWinningNeurons)(
            base_directory, s_features, n_neurons, m_neurons, 
            tree_extension = 'csv') for k in file_names )
        
def CreateGValueFluxTables(
    base_directory,
    n_neurons,
    m_neurons,
    tree_extension = 'csv'
):

    tree_directory = base_directory + '/trees/' 
    file_names = sorted(Path(tree_directory).glob("*."+tree_extension))
    par1 = 'gval'
    par2 = 'flux'
    sum_table = pd.DataFrame({
                    'id': [],
                    'winning_Nx': [],
                    'winning_Ny': [],
                    'sum_'+par1:[], 
                    'sum_'+par2:[]}, index=[])
    ext = ('.csv')
    a = 0
    for file in tqdm(file_names):
        filename = file.stem
        attributes = pd.read_csv(base_directory 
                                 + '/trees/'+filename+ext, sep=',', header=0, low_memory=False) 
        winning_neurons = pd.read_csv(base_directory 
                                      + '/SOMs/'+filename+ext, sep=',', header=0, low_memory=False) 
        flux = pd.DataFrame({
                        'winning_Nx': [],
                        'winning_Ny': [],
                        'sum_par1':[],
                        'sum_par2':[]}, index=[])
        img_par1 = np.zeros(n_neurons*m_neurons).reshape((n_neurons, m_neurons))
        img_par2 = np.zeros(n_neurons*m_neurons).reshape((n_neurons, m_neurons))

        for nx in range(0,n_neurons):
            for ny in range(0,m_neurons):
                mask1 = winning_neurons['winning_Nx'] == nx
                mask2 = winning_neurons['winning_Ny'] == ny

                selected1 = attributes[mask1&mask2][' '+par1]
                selected2 = attributes[mask1&mask2][' '+par2]

                selected1 = np.nan_to_num(selected1)
                selected2 = np.nan_to_num(selected2)

                selected1 = selected1.astype(float)
                selected2 = selected2.astype(float)

                sum_par1= selected1.sum(axis = 0)
                sum_par2= selected2.sum(axis = 0)

                img_par1[n_neurons-1-ny,nx] = sum_par1
                img_par2[n_neurons-1-ny,nx] = sum_par2
                
                tmp =  pd.DataFrame({
                    'id': filename,
                    'winning_Nx': nx,
                    'winning_Ny': ny,
                    'sum_'+par1: sum_par1,  
                    'sum_'+par2: sum_par2}, index=[a])
                a = a + 1
                sum_table = pd.concat([sum_table,tmp], ignore_index=True)

        fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.))

        im1 = ax[0].imshow((img_par1), extent=[0,n_neurons,0,m_neurons])
        ax[0].set_xlabel('Neuron_x')
        ax[0].set_ylabel('Neuron_y')
        ax[0].set_title(par1)
        cbar1 = plt.colorbar(im1, ax=ax[0])
        cbar1.set_label(par1)
        cbar1.formatter.set_powerlimits((0, 0))

        im2 = ax[1].imshow((img_par2), extent=[0,n_neurons,0,m_neurons])
        cbar2 = plt.colorbar(im2, ax=ax[1])
        cbar2.set_label(par2)
        cbar2.formatter.set_powerlimits((0, 0))
        ax[1].set_xlabel('Neuron_x')
        ax[1].set_ylabel('Neuron_y')
        ax[1].set_title(par2)

        os.makedirs(base_directory + '/sum_tables/', exist_ok=True)
        fig.suptitle(filename)
        fig.subplots_adjust(left=0.08, bottom=0.13, right=0.95, top=0.9, wspace=0.26, hspace=0.1)
        fig.savefig(base_directory + '/sum_tables/'+ filename +'_summary.pdf')
        fig.clf()
    os.makedirs(base_directory + '/sum_tables/', exist_ok=True)        
    sum_table.to_csv(base_directory + '/sum_tables/sum_table.csv')
    
    
def get_stats(array):
    array = np.ma.masked_invalid(array)
    return np.sum(array), np.mean(array), np.std(array)

def calculate_normalised_excess(iid, sum_table, PS_stats, par, nx, ny):
    xmask = PS_stats['winning_Nx'] == nx
    ymask = PS_stats['winning_Ny'] == ny
    avg = PS_stats[xmask&ymask]['avg_'+par].values
    std = PS_stats[xmask&ymask]['std_'+par].values 
    
    idmask = sum_table['id'] == iid
    xxmask = sum_table['winning_Nx'] == nx
    yymask = sum_table['winning_Ny'] == ny
    
    sum_par = sum_table[idmask&xxmask&yymask]['sum_'+par].values
    np.seterr(invalid='ignore')
    try:
        excess = np.divide((sum_par - avg), std)
    except: 
        excess =  0
    return excess


def GetPatternSpectra(
    base_directory,
    n_neurons,
    m_neurons,
    tree_extension = 'csv'    
):
    par1 = 'gval'
    par2 = 'flux'

    flux = pd.DataFrame({
                    'winning_Nx': [],
                    'winning_Ny': [],
                    'sum_'+par1:[], 
                    'avg_'+par1:[],
                    'std_'+par1:[], 
                    'sum_'+par2:[],
                    'avg_'+par2:[], 
                    'std_'+par2:[],
    }, index=[])

    img1_avg = np.zeros(n_neurons*m_neurons).reshape((n_neurons, m_neurons))
    img1_std = np.zeros(n_neurons*m_neurons).reshape((n_neurons, m_neurons))
    img2_avg = np.zeros(n_neurons*m_neurons).reshape((n_neurons, m_neurons))
    img2_std = np.zeros(n_neurons*m_neurons).reshape((n_neurons, m_neurons))

    for nx in range(0,n_neurons):
        for ny in range(0,m_neurons):
            directory = base_directory + '/sum_tables/'
            f = os.path.join(directory, 'sum_table.csv')

            if os.path.isfile(f):
                sum_table = pd.read_csv(f)
                xmask = sum_table['winning_Nx'] == nx
                ymask = sum_table['winning_Ny'] == ny
                selected_par1 = sum_table[xmask&ymask]['sum_'+par1]
                selected_par2 = sum_table[xmask&ymask]['sum_'+par2]
                sum1, avg1, std1 = get_stats(selected_par1)
                sum2, avg2, std2 = get_stats(selected_par2)

                tmp = pd.DataFrame({
                    'winning_Nx': nx,
                    'winning_Ny': ny,
                    'sum_'+par1: sum1, 
                    'avg_'+par1: avg1,
                    'std_'+par1: std1, 
                    'sum_'+par2: sum2,
                    'avg_'+par2: avg2, 
                    'std_'+par2: std2,
                },
                index=[nx*n_neurons + ny])
            else:
                print("sum table not found!")
                break
                
            img1_avg[n_neurons-1-ny,nx] = avg1
            img1_std[n_neurons-1-ny,nx] = std1
            img2_avg[n_neurons-1-ny,nx] = avg2
            img2_std[n_neurons-1-ny,nx] = std2
            flux = pd.concat([flux, tmp], ignore_index=True) 

    flux.to_csv(directory+'PS_stats.csv') 
    
    
def GetNormalisedExcess(
    base_directory,
    n_neurons,
    m_neurons,
    tree_extension = 'csv'    
):
    par1 = 'gval'
    par2 = 'flux'
    tree_directory = base_directory + '/trees/' 
    file_names = sorted(Path(tree_directory).glob("*."+tree_extension))

    excess = pd.DataFrame({
                    'id': [],
                    'winning_Nx': [],
                    'winning_Ny': [],
                    'excess_'+par1:[], 
                    'excess_'+par2:[],

    }, index=[])

    sum_table = pd.read_csv(base_directory + '/sum_tables/sum_table.csv')
    PS_stats = pd.read_csv(base_directory + '/sum_tables/PS_stats.csv')

    a=0
    for file in tqdm(file_names):
        filename = file.stem
        SOMfile = (base_directory + '/SOMs/'+filename+'SOMs.csv')

        if os.path.exists(file):
            fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
            img1 = np.zeros(n_neurons*m_neurons).reshape((n_neurons, m_neurons))
            img2 = np.zeros(n_neurons*m_neurons).reshape((n_neurons, m_neurons))

            for nx in range(0,n_neurons):
                for ny in range(0,m_neurons):
                    directory = base_directory + '/excess_tables/'
                    os.makedirs(directory, exist_ok=True)
                    excess1 = calculate_normalised_excess(filename, sum_table, PS_stats, par1, nx, ny)
                    excess2 = calculate_normalised_excess(filename, sum_table, PS_stats, par2, nx, ny)

                    tmp = pd.DataFrame({
                        'id': filename,
                        'winning_Nx': nx,
                        'winning_Ny': ny,
                        'excess_'+par1: excess1, 
                        'excess_'+par2: excess2,
                    },
                    index=[a])
                    a = a+1

                    img1[n_neurons-1-ny,nx] = excess1
                    img2[n_neurons-1-ny,nx] = excess2
                    excess = pd.concat([excess, tmp], ignore_index=True) 

            im1 = ax[0].imshow(img1, extent=[0,n_neurons,0,m_neurons], vmax = 3, vmin = -3, cmap='RdBu_r')
            im2 = ax[1].imshow(img2, extent=[0,n_neurons,0,m_neurons], vmax = 3, vmin = -3, cmap='RdBu_r')

            ax[0].set_xlabel('Neuron_x')
            ax[0].set_ylabel('Neuron_y')
            ax[0].set_title(par1)
            cbar1 = plt.colorbar(im1, ax=ax[0])
            cbar1.set_label('normalised excess')

            ax[1].set_xlabel('Neuron_x')
            ax[1].set_ylabel('Neuron_y')
            ax[1].set_title(par2)
            cbar2 = plt.colorbar(im2, ax=ax[1])
            cbar2.set_label('normalised excess')
            fig.suptitle(filename)
            fig.subplots_adjust(left=0.08, bottom=0.13, right=0.95, top=0.9, wspace=0.26, hspace=0.1)

            fig.savefig(directory+'/'+filename+'_normalised_excess.pdf')
            fig.clf()
        else:
            continue
    excess.to_csv(directory+'excess.csv') 
    
    
def PlotNormalisedSelfOrganisedPS(
    base_directory,
    n_neurons,
    m_neurons,
    attribute = 'flux', # flux or gval 
    tree_extension = 'csv'  
):
    
    file_names = sorted(Path(base_directory+'/trees/').glob("*."+tree_extension))
    n_x = round(np.sqrt(len(file_names)))
    nfigs = len(file_names)
    n_y = round(nfigs / n_x)

    excess_table = pd.read_csv(base_directory+'/excess_tables/excess.csv',
                           sep=',', header=0, low_memory=False)
    cubes = []
    for file in file_names:
        par = attribute
        img1 = np.zeros(n_neurons*m_neurons).reshape((n_neurons, m_neurons))
        for nx in range(0,n_neurons):
            for ny in range(0,m_neurons):
                mask_id = excess_table['id'] == file.stem
                mask_x = excess_table['winning_Nx'] == nx
                mask_y = excess_table['winning_Ny'] == ny
                img1[n_neurons-1-ny,nx] = excess_table[mask_id&mask_x&mask_y]['excess_'+par].values[0]
        cubes.append(img1)
    
    kwargs = {"extent": (0,n_neurons,0,m_neurons), "vmin": -3, "vmax": 3, "cmap":'RdBu_r'}

    fig, axes = plt.subplots(n_y, n_x, figsize=(8, 6), 
                             sharex=True, sharey=True, gridspec_kw={"wspace": 0.03, "hspace": 0.3})
    for ii, axis in enumerate(axes.ravel()):
        #print(ii)
        if ii < nfigs :
            name = file_names[ii].stem
            im = axis.imshow(cubes[ii], **kwargs)
            axis.set_title(name, fontsize=9)
            if ii % n_x == 0 : 
                axis.set_ylabel('N_y')
            if ii // n_x == n_y - 1 : 
                axis.set_xlabel('N_x')
                
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle('normalised ' + attribute + ' pattern spectra')
    
    
def PlotExcessNeurons(
    base_directory,
    n_neurons,
    m_neurons,
    attribute = 'flux', # flux or gval 
    upper_limit = 2.0, # sigma
    lower_limit = 2.0, # sigma
    tree_extension = 'csv'  
):
    
    file_names = sorted(Path(base_directory+'/trees/').glob("*."+tree_extension))
    n_x = round(np.sqrt(len(file_names)))
    nfigs = len(file_names)
    n_y = round(nfigs / n_x)

    excess_table = pd.read_csv(base_directory+'/excess_tables/excess.csv',
                           sep=',', header=0, low_memory=False)
    cubes = []
    for file in file_names:
        par = attribute
        img1 = np.zeros(n_neurons*m_neurons).reshape((n_neurons, m_neurons))
        for nx in range(0,n_neurons):
            for ny in range(0,m_neurons):
                mask_id = excess_table['id'] == file.stem
                mask_x = excess_table['winning_Nx'] == nx
                mask_y = excess_table['winning_Ny'] == ny            
                img1[n_neurons-1-ny,nx] = excess_table[mask_id&mask_x&mask_y]['excess_'+par].values[0]       
        m1 = ma.masked_where(img1 <= upper_limit, img1).mask
        m2 = ma.masked_where(img1 >= -1 * np.abs(lower_limit), img1).mask
        new = ma.masked_array(img1, m1 & m2)
        cubes.append(new)
        
    kwargs = {"extent": (0,n_neurons,0,m_neurons), "vmin": -3, "vmax": 3, "cmap":'RdBu_r'}

    fig, axes = plt.subplots(n_y, n_x, figsize=(8, 6), 
                             sharex=True, sharey=True, gridspec_kw={"wspace": 0.03, "hspace": 0.3})
    for ii, axis in enumerate(axes.ravel()):
        if ii < nfigs :
            name = file_names[ii].stem
            im = axis.imshow(cubes[ii], **kwargs)
            axis.set_title(name, fontsize=9)
            if ii % n_x == 0 : 
                axis.set_ylabel('N_y')
            if ii // n_x == n_y - 1 : 
                axis.set_xlabel('N_x')
                
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle('excess neurons in ' + attribute + ' pattern spectra')
