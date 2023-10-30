from pathlib import Path
from tqdm.notebook import tqdm
from minisom import MiniSom
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from joblib import Parallel, delayed
from joblib_progress import joblib_progress
from multiprocessing import Pool

import os
import pandas as pd
import minisom 
import pickle
import random
import time

from draco.core import containers
import astropy
from astropy.io import fits


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
    
def CleanupBaseDirectory(
    base_directory,
    disccofanSOM_directory,
):
    command = 'rm -r ' + base_directory
    os.system('rm -r ' + disccofanSOM_directory + '/SOMs/')
    os.makedirs(base_directory, exist_ok=True)
    for folder in ['/SOMs/', '/filtered/', '/excess_tables/', '/sum_tables/','/plots/']:
        os.system(command + folder)
        
def CleanupTemp(
    disccofanSOM_directory,
):
    os.system('rm -r ' + disccofanSOM_directory + '/SOMs/*.csv')

        
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
    
def CreateDataIDs_forCHIME(
    revision = "rev_07",
):
    daily_processing_path = Path(f"/project/rpp-chime/chime/chime_processed/daily/{revision}/")
    obs_set = list(daily_processing_path.iterdir())
    id_set = []
    for obs in obs_set:
        s = str(obs)
        try: 
            id_set.append(int(s[-4:]))
        except:
            continue
    return sorted(id_set)

def CreateFitsforCHIME(
    ids, 
    base_directory: str,
    image_directory,
    revision = "rev_07",
):
    for ii, csd in tqdm(enumerate(ids)):
        csd = str(csd)
        daily_processing_path = Path(f"/project/rpp-chime/chime/chime_processed/daily/{revision}/")
    
        try:
            fits_file = Path( image_directory + '/'+csd+'.fits' )
            my_abs_path = fits_file.resolve(strict=True)
            
        except FileNotFoundError:
            # tree doesn't exist    
            try: 
                name = "delayspectrum_lsd_" + csd + ".h5"
                ds = containers.DelaySpectrum.from_file( daily_processing_path / csd / name)
                newimage = lscale(ds.spectrum[:].T)
                os.makedirs( image_directory , exist_ok=True)
                astropy.io.fits.writeto( image_directory + '/'+csd+'.fits', newimage, header=None,) 
            except:
                print("Failed to create a fits file!")

def CreateTrees(
    imagefiles,
    extension: str,
    disccofan_directory: str,
    base_directory: str,
    tree_directory,
    n_thread = 1, 
    n_connectivity = 8,
    n_attributes = 12,
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
    os.makedirs(tree_directory, exist_ok=True) 
    disccofan_directory = disccofan_directory + '/disccofan'
    for file in tqdm(imagefiles):
        filename = file.stem
        filedir = os.path.splitext(file)[0]
        outname = tree_directory + '/' + filename
        command = '%s -g 1,1,1 --threads %d  -c %d  -a %d  --inprefix %s --intype %s -o treeCSV --outprefix %s -l %d'%(disccofan_directory, n_thread, n_connectivity, n_attributes, filedir, extension, outname, lval)
        try: 
            os.system(command)
        except:
            print("Failed to create a tree..!")

def CreateSingleTree(
    imagefile,
    extension: str,
    disccofan_directory: str,
    base_directory: str,
    tree_directory,
    n_thread = 1, 
    n_connectivity = 8,
    n_attributes = 12,
    lval = 100,
):
    os.makedirs(tree_directory, exist_ok=True) 
    disccofan_directory = disccofan_directory + '/disccofan'

    filename = Path(imagefile).stem
    filedir = os.path.splitext(Path(imagefile))[0]
    outname = tree_directory + '/' + filename
    command = '%s -g 1,1,1 --threads %d  -c %d  -a %d  --inprefix %s --intype %s -o treeCSV --outprefix %s -l %d'%(disccofan_directory, n_thread, n_connectivity, n_attributes, filedir, extension, outname, lval)
    try:
        tree_file = Path(outname + '.csv')
        my_abs_path = tree_file.resolve(strict=True)
        print("Tree exists for " + filename)
    except FileNotFoundError:
        # tree doesn't exist    
        try: 
            os.system(command)
        except:
            print("Failed to create a tree..!")
            
            
            
def CreateTrees_in_Parallel(
    imagefiles,
    extension: str,
    disccofan_directory: str,
    base_directory: str,
    tree_directory: str,
    n_thread = 1, 
    n_connectivity = 8,
    n_attributes = 12,
    lval = 100,
):    
    with joblib_progress("Creating trees..."):
        element_run = Parallel(n_jobs=-1)(delayed(CreateSingleTree)(
            image,
            extension,
            disccofan_directory,
            base_directory,
            tree_directory,
            n_thread, 
            n_connectivity,
            n_attributes, 
            lval,) for image in imagefiles )        
            
def ShowImages(
    imagefiles,
    base_directory,
):
    # setting values to rows and column variables
    rows = round(np.sqrt(len(imagefiles)))
    columns = round(np.sqrt(len(imagefiles)))
    
    fig, axs = plt.subplots(rows, columns, figsize=(10, 8) )
    fig.subplots_adjust(hspace = .1, wspace=.05)
    axs = axs.ravel()
          
    for ii, ax in enumerate(axs):
        if ii < len(imagefiles):
            img_i = imagefiles[ii]
            ax.imshow(mpimg.imread(img_i), cmap='Greys_r') #cv2.imread(img_i))
            ax.axis('off')
            ax.set_title(img_i.stem)
        else:
            ax.axis('off')
    os.makedirs(base_directory + '/plots/', exist_ok=True)
    fig.tight_layout()
    plt.savefig(base_directory + '/plots/' + 'original_img.pdf')
    plt.show()
    
    
def LoadTrees(
    base_directory : str,
    tree_directory,
    extension = "csv", 
    tree_selecting_rate = 0.1,
    sampling_rate = 1.0,  
):
    treefiles = sorted(Path(tree_directory+'/').glob("*."+extension))
    if tree_selecting_rate >= 1.0:
        selected = treefiles
    else:
        selected = np.random.choice(treefiles, size=round(len(treefiles)*tree_selecting_rate) )
    data = pd.DataFrame()    
    
    for tree_file in tqdm(selected):
        tmp = pd.read_csv(tree_file, sep=',', header=0, low_memory=False #).sample(frac = sampling_rate) # --> too slow
                          , skiprows=lambda i: i>0 and random.random() > sampling_rate) 
        tmp = trim_data(tmp)     
        #sampled_tmp = tmp.sample(n = round(len(tmp.index)*sampling_rate) )
        data = pd.concat([data, tmp], ignore_index=True)
    print(str(len(data)) + ' nodes are loaded in total!')
    return data

def LoadSingleTree(
    treefile,
    base_directory : str,
    extension = "csv", 
    sampling_rate = 1.0, 
):
    treefile = Path(treefile)
    data = pd.DataFrame()    
    tmp = pd.read_csv(treefile, sep=',', header=0, low_memory=False) 
    tmp = trim_data(tmp)     
    sampled_tmp = tmp.sample(n = round(len(tmp.index)*sampling_rate) )
    data = pd.concat([data, sampled_tmp], ignore_index=True)
    return data

def LoadTrees_in_Parallel(
    treefile,
    base_directory : str,
    extension = "csv", 
    sampling_rate = 1.0,
):
    data = pd.DataFrame()
    with joblib_progress("Creating trees..."):
        element_run = Parallel(n_jobs=-1)(delayed(LoadSingleTree)(
        treefile,
        base_directory,
        extension = "csv",
        sampling_rate = 1.0,) for image in imagefiles )  
        data = pd.concat([data, sampled_tmp], ignore_index=True)
    
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
    plt.tight_layout()
    plt.savefig(base_directory + '/SOM_distance_map.pdf')
    with open(base_directory+'/som.p', 'wb') as outfile:
        pickle.dump(som, outfile)
    print('Training completed!')
    
    
def GetWinningNeurons(
    base_directory,
    tree_directory,
    filename,
    s_features,
    n_neurons,
    m_neurons,
    tree_extension = 'csv',
):    
    with open(base_directory+'/som.p', 'rb') as infile:
        som = pickle.load(infile)    

    tree = Path(tree_directory + '/' + filename + '.' + tree_extension)
    data = pd.read_csv(tree, sep=',', header=0, low_memory=False) 
    data = trim_data(data)
    features = np.array(list(data.keys()))

    n_columns = np.shape(data)[-1]
    flat_data = []

    for i in range(0,len(features)):
        flat_data.append(data.iloc[:,i].to_numpy())

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
    tree_directory,
    s_features,
    n_neurons,
    m_neurons,
    tree_extension = 'csv',
):

    tree_files = sorted(Path(tree_directory).glob("*."+tree_extension))
    
    with joblib_progress("Calculating winning neurons..."):
        element_run = Parallel(n_jobs=-1)(delayed(GetWinningNeurons)(
            base_directory, tree_directory, tree.stem, s_features, n_neurons, m_neurons, 
            tree_extension = 'csv') for tree in tree_files )
        
def CreateGValueFluxTables(
    base_directory,
    tree_directory,
    n_neurons,
    m_neurons,
    tree_extension = 'csv',
):

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
        attributes = pd.read_csv(tree_directory 
                                 + '/'+filename+ext, sep=',', header=0, low_memory=False) 
        winning_neurons = pd.read_csv(base_directory 
                                      + '/SOMs/'+filename+ext, sep=',', header=0, low_memory=False) 
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
                    'id': str(filename),
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
        fig.tight_layout()
        fig.savefig(base_directory + '/sum_tables/'+ filename +'_summary.pdf')
        fig.clf()
    os.makedirs(base_directory + '/sum_tables/', exist_ok=True)        
    sum_table.to_csv(base_directory + '/sum_tables/sum_table.csv')


def CreateGValueFluxTables_Single(
    base_directory,
    tree_directory,
    treefile,
    n_neurons,
    m_neurons,
    tree_extension = 'csv',
):
    par1 = 'gval'
    par2 = 'flux'
    sum_table = pd.DataFrame({
                    'id': [],
                    'winning_Nx': [],
                    'winning_Ny': [],
                    'sum_'+par1:[], 
                    'sum_'+par2:[]}, index=[])
    
    ext = '.' + tree_extension

    filename = Path(treefile).stem
    attributes = pd.read_csv(tree_directory 
                             + '/'+filename+ext, sep=',', header=0, low_memory=False) 
    winning_neurons = pd.read_csv(base_directory 
                                  + '/SOMs/'+filename+ext, sep=',', header=0, low_memory=False) 
    
    img_par1 = np.zeros(n_neurons*m_neurons).reshape((n_neurons, m_neurons))
    img_par2 = np.zeros(n_neurons*m_neurons).reshape((n_neurons, m_neurons))

    a = 0
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
                'id': str(filename),
                'winning_Nx': nx,
                'winning_Ny': ny,
                'sum_'+par1: sum_par1,  
                'sum_'+par2: sum_par2}, index=[a])
            a = a + 1
            sum_table = pd.concat([sum_table,tmp], ignore_index=True)
            
    os.makedirs(base_directory + '/sum_tables/tables/', exist_ok=True)
    sum_table.to_csv(base_directory + '/sum_tables/tables/'+filename+'.csv')
    
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

    os.makedirs(base_directory + '/sum_tables/summaries/', exist_ok=True)
    fig.suptitle(filename)
    fig.subplots_adjust(left=0.08, bottom=0.13, right=0.95, top=0.9, wspace=0.26, hspace=0.1)
    fig.tight_layout()
    fig.savefig(base_directory + '/sum_tables/summaries/'+ filename +'_summary.pdf')
    fig.clf()
    
    
def CreateGValueFluxTables_in_Parallel(
    base_directory,
    tree_directory,
    n_neurons,
    m_neurons,
    tree_extension = 'csv',
):
    tree_files = sorted(Path(tree_directory).glob("*."+tree_extension))
    
    with joblib_progress("Creating gvalue and flux tables..."):
        element_run = Parallel(n_jobs=-1)(delayed(CreateGValueFluxTables_Single)(
            base_directory, tree_directory, treefile, n_neurons, m_neurons, 
            tree_extension = 'csv') for treefile in tree_files )

        
def my_read_csv(filename):
    return pd.read_csv(filename, sep=',', header=0, low_memory=False) 

def load_csvs(file_list):
    # set up your pool
    pool = Pool() 
    df_list = pool.map(my_read_csv, file_list)
    # reduce the list of dataframes to a single dataframe
    return pd.concat(df_list, ignore_index=True)


def CreateSumTable(
    base_directory,
):  
    try:
        table_files = sorted(Path(base_directory + '/sum_tables/tables/').glob("*.csv"))
        sum_table = load_csvs(table_files)
        sum_table.to_csv(base_directory + '/sum_tables/sum_table.csv')
    except:
        print("No tables to sum. Create tables first!")
    
def get_stats(array):
    array = np.ma.masked_invalid(array)
    return np.sum(array), np.mean(array), np.std(array)

def calculate_normalised_excess(iid, sum_table, PS_stats, par, nx, ny):
    xmask = PS_stats['winning_Nx'] == nx
    ymask = PS_stats['winning_Ny'] == ny
    avg = PS_stats[xmask&ymask]['avg_'+par].values
    std = PS_stats[xmask&ymask]['std_'+par].values 
    
    sum_table["id"]=sum_table["id"].values.astype(str)    
    idmask = sum_table['id'] == str(iid)
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
    tree_directory,
    n_neurons,
    m_neurons,
    tree_extension = 'csv',
):
    par1 = 'gval'
    par2 = 'flux'

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
                        'id': str(filename),
                        'winning_Nx': nx,
                        'winning_Ny': ny,
                        'excess_'+par1: excess1, 
                        'excess_'+par2: excess2,
                    },
                    index=[a])
                    a = a+1
                    print(tmp)
                    print(excess)
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
            fig.tight_layout()
            fig.savefig(directory+'/'+filename+'_normalised_excess.pdf')
            fig.clf()
        else:
            continue
    excess.to_csv(directory+'excess.csv') 
    
def GetNormalisedExcess_Single(
    base_directory,
    file,
    n_neurons,
    m_neurons,
    tree_extension = 'csv'    
):
    par1 = 'gval'
    par2 = 'flux'

    sum_table = pd.read_csv(base_directory + '/sum_tables/sum_table.csv')
    PS_stats = pd.read_csv(base_directory + '/sum_tables/PS_stats.csv')

    a=0
    filename = Path(file).stem
    excess = pd.DataFrame({
            'id': [],
            'winning_Nx': [],
            'winning_Ny': [],
            'excess_'+par1:[], 
            'excess_'+par2:[],

    }, index=[])

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
                    'id': str(filename),
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
        fig.tight_layout()
        fig.savefig(directory+'/'+filename+'_normalised_excess.pdf')
        fig.clf()
        
    directory = base_directory + '/excess_tables/tables/'
    os.makedirs(directory, exist_ok=True)
    excess.to_csv(directory+str(filename)+'.csv') 

def GetNormalisedExcess_in_Parallel(
    base_directory,
    tree_directory,
    n_neurons,
    m_neurons,
    tree_extension = 'csv',
):
    tree_files = sorted(Path(tree_directory).glob("*."+tree_extension))
    
    with joblib_progress("Creating normalised excess..."):
        element_run = Parallel(n_jobs=-1)(delayed(GetNormalisedExcess_Single)(
            base_directory, treefile, n_neurons, m_neurons,
            tree_extension = 'csv') for treefile in tree_files )
        
def CombineExcessTables(
    base_directory,
):
    try:
        table_files = sorted(Path(base_directory + '/excess_tables/tables/').glob("*.csv"))
        excess_table = load_csvs(table_files)
        excess_table.to_csv(base_directory + '/excess_tables/excess.csv')
    except:
        print("No tables to sum. Create tables first!")

        
def PlotNormalisedSelfOrganisedPS_in_Parallel(
    base_directory,
    tree_directory,
    n_neurons,
    m_neurons,
    attribute = 'flux', # flux or gval 
    tree_extension = 'csv',
):
    file_names = sorted(Path(tree_directory+'/').glob("*."+tree_extension))
    n_x = round(np.sqrt(len(file_names)))
    nfigs = len(file_names)
    n_y = round(nfigs / n_x)

    excess_table = pd.read_csv(base_directory+'/excess_tables/excess.csv',
                           sep=',', header=0, low_memory=False)
    excess_table["id"]=excess_table["id"].values.astype(str)
    cubes = []
    for file in file_names:
        par = attribute
        img1 = np.zeros(n_neurons*m_neurons).reshape((n_neurons, m_neurons))
        for nx in range(0,n_neurons):
            for ny in range(0,m_neurons):
                mask_id = excess_table['id'] == str(file.stem)
                mask_x = excess_table['winning_Nx'] == nx
                mask_y = excess_table['winning_Ny'] == ny
                img1[n_neurons-1-ny,nx] = excess_table[mask_id&mask_x&mask_y]['excess_'+par].values[0]
        cubes.append(img1)
    
    with joblib_progress("Creating normalised self organising PS..."):
        element_run = Parallel(n_jobs=-1)(delayed(GetNormalisedExcess_Single)(
            base_directory, treefile, n_neurons, m_neurons,
            tree_extension = 'csv') for treefile in tree_files )
    

def PlotExcessPS(
    base_directory,
    file_names,
    n_neurons,
    m_neurons,
    attribute = 'flux', # flux or gval 
    tree_extension = 'csv'  
):
    n_x = 5
    nfigs = len(file_names)
    n_y = 6

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
    fig.suptitle('Normalised ' + attribute + ' pattern spectra')
    os.makedirs(base_directory + '/plots/', exist_ok=True)
    fig.tight_layout()
    fig.savefig(base_directory + '/plots/' + 'normalised_' + attribute + '_ps' + '.pdf')   
    
        
def PlotNormalisedSelfOrganisedPS(
    base_directory,
    tree_directory,
    n_neurons,
    m_neurons,
    attribute = 'flux', # flux or gval 
    tree_extension = 'csv',
):
    
    file_names = sorted(Path(tree_directory+'/').glob("*."+tree_extension))
    n_x = round(np.sqrt(len(file_names)))
    nfigs = len(file_names)
    n_y = round(nfigs / n_x)

    excess_table = pd.read_csv(base_directory+'/excess_tables/excess.csv',
                           sep=',', header=0, low_memory=False)
    excess_table["id"]=excess_table["id"].values.astype(str)
    cubes = []
    for file in file_names:
        par = attribute
        img1 = np.zeros(n_neurons*m_neurons).reshape((n_neurons, m_neurons))
        for nx in range(0,n_neurons):
            for ny in range(0,m_neurons):
                mask_id = excess_table['id'] == str(file.stem)
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
    fig.suptitle('Normalised ' + attribute + ' pattern spectra')
    os.makedirs(base_directory + '/plots/', exist_ok=True)
    fig.tight_layout()
    fig.savefig(base_directory + '/plots/' + 'normalised_' + attribute + '_ps' + '.pdf')    
    
    
def PlotExcessNeurons(
    base_directory,
    tree_directory,
    n_neurons,
    m_neurons,
    upper_limit, # sigma
    lower_limit, # sigma
    attribute = 'flux', # flux or gval 
    tree_extension = 'csv',
):
    
    file_names = sorted(Path(tree_directory+'/').glob("*."+tree_extension))
    n_x = round(np.sqrt(len(file_names)))
    nfigs = len(file_names)
    n_y = round(nfigs / n_x)

    excess_table = pd.read_csv(base_directory+'/excess_tables/excess.csv',
                           sep=',', header=0, low_memory=False)
    excess_table["id"]=excess_table["id"].values.astype(str)

    cubes = []
    for file in file_names:
        par = attribute
        img1 = np.zeros(n_neurons*m_neurons).reshape((n_neurons, m_neurons))
        for nx in range(0,n_neurons):
            for ny in range(0,m_neurons):
                mask_id = excess_table['id'] == str(file.stem)
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
                
    fig.subplots_adjust(left=0.08, bottom=0.13, right=0.95, top=0.9, wspace=0.26, hspace=0.1)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle('Excess neurons in ' + attribute + ' pattern spectra')
    os.makedirs(base_directory + '/plots/', exist_ok=True)
    fig.tight_layout()
    fig.savefig(base_directory + '/plots/' + 'excess_neurons_' + attribute + '_ps' + '.pdf')    

def FilterImageFromNeuron(
    filename,
    base_directory,
    image_directory,
    disccofanSOM_directory,
    n_neurons,
    m_neurons,
    nx, 
    ny,
    n_connectivity = 8,
    n_attributes = 12,
    image_extension = 'jpg',
):
    somfile = base_directory + '/SOMs/' + filename + '.csv'
    copy_directory = disccofanSOM_directory + '/SOMs/'
    os.makedirs(copy_directory, exist_ok=True)
    
    copied_somfile = './SOMs/' + filename + '.csv'
    outdir = base_directory + '/filtered/' + filename + '/'
    outname = outdir + str(nx) + ',' + str(ny)
    os.makedirs(outdir, exist_ok=True)
    
    path_to_file = outname + '.' + image_extension
    path = Path(path_to_file)

    if path.is_file():
        print("Filtered image already exists!")
    else:     
        command1 = 'cp -p '+ somfile + ' ' + copy_directory
        command2 = '%s/disccofan -g 1,1,1 --threads 1  -c %d  -a %d \
        --inprefix %s --intype %s -o som --somsize %d --somfile %s --somneuron %d,%d \
        --outprefix %s --outtype %s'%(disccofanSOM_directory, n_connectivity, n_attributes, image_directory + '/' + filename, image_extension, n_neurons, copied_somfile, nx, ny, outname, image_extension)

        try: 
            sompathfile = disccofanSOM_directory + '/SOMs/' + filename + '.csv'
            sompath = Path(sompathfile)
            if path.is_file():
                os.system(command2)
            else:     
                os.system(command1+" && "+command2)

        except:
            print("Image filtering failed.....!!")
            
            
def GetExcessNeurons(
    base_directory,
    tree_directory,
    n_neurons,
    m_neurons,
    upper_limit, # sigma
    lower_limit, # sigma
    attribute = 'flux', # flux or gval 
    tree_extension = 'csv',
):
    
    file_names = sorted(Path(tree_directory+'/').glob("*."+tree_extension))
    excess_table = pd.read_csv(base_directory+'/excess_tables/excess.csv',
                           sep=',', header=0, low_memory=False)
    excess_table["id"]=excess_table["id"].values.astype(str)

    cubes = []
    for file in file_names:
        par = attribute
        neurons = []
        for nx in range(0,n_neurons):
            for ny in range(0,m_neurons):
                mask_id = excess_table['id'] == str(file.stem)
                mask_x = excess_table['winning_Nx'] == nx
                mask_y = excess_table['winning_Ny'] == ny    
                tmp = excess_table[mask_id&mask_x&mask_y]['excess_'+par].values[0]
                
                if tmp == np.nan:
                    continue
                elif tmp >= upper_limit or tmp <= -1*np.abs(lower_limit) :
                    neurons.append([nx,ny])
        cubes.append(neurons) 
    return cubes
            

def FilterAllExcessNeurons(
    base_directory,
    image_directory,
    disccofanSOM_directory,
    n_neurons,
    m_neurons,
    upper_limit, # sigma
    lower_limit, # sigma
    attribute = 'flux',
    tree_extension = 'csv',
    n_connectivity = 8,
    n_attributes = 12,
    image_extension = 'jpg',
):
    excess_neurons = GetExcessNeurons(
    base_directory, n_neurons, m_neurons, upper_limit, lower_limit, attribute, tree_extension,)
    imagefiles = sorted( Path( image_directory + '/' ).glob("*."+image_extension))
    CleanupTemp(disccofanSOM_directory)
    
    for ii, neurons in enumerate(excess_neurons):    
        filename = imagefiles[ii].stem
        if len(neurons) != 0 : 
            with joblib_progress("Filtering "+ filename):
                element_run = Parallel(n_jobs=-1)(delayed(FilterImageFromNeuron)(
                    filename,
                    base_directory,
                    disccofanSOM_directory,
                    n_neurons,
                    m_neurons,
                    nx = neuron[0], 
                    ny = neuron[1],
                    n_connectivity = n_connectivity,
                    n_attributes = n_attributes,
                    image_extension = image_extension) for neuron in neurons )
    print('Filtering completed!')
    
def PlotExcessNeuronsFilteredImage(
    base_directory,
    image_directory,
    n_neurons,
    m_neurons,
    upper_limit, 
    lower_limit, 
    attribute = 'flux',
    tree_extension = 'csv',
    image_extension = 'jpg',
):

    excess_neurons = GetExcessNeurons(
    base_directory, n_neurons, m_neurons, upper_limit, lower_limit, attribute, tree_extension,)
        
    filenames = sorted(Path( image_directory + '/' ).glob("*."+image_extension))
    
    for ii, neurons in enumerate(excess_neurons):    
        filename = filenames[ii].stem
        if len(neurons) != 0 :        
            columns = round(np.sqrt(len(filenames)))
            rows = round(len(neurons) / columns + 0.499)
            
            fig, axs = plt.subplots(rows, columns, figsize=(3.3 * columns, 2.7 * rows) )
            fig.subplots_adjust(left=0.08, bottom=0.13, right=0.95, top=0.9, wspace=0.26, hspace=0.1)
            axs = axs.ravel()
            
            for ii, ax in enumerate(axs):
                if ii < len(neurons):
                    neuron = neurons[ii]
                    img_i = base_directory + '/filtered/' + filename + '/' + str(neuron[0]) + ',' + str(neuron[1]) +'.'+image_extension
                    ax.imshow(mpimg.imread(img_i), cmap='Greys_r') #cv2.imread(img_i))
                    ax.axis('off')
                    ax.set_title('(' + Path(img_i).stem + ')' )
                else:
                    ax.axis('off')
                plt.suptitle(filename)
            os.makedirs(base_directory + '/plots/', exist_ok=True)
            fig.tight_layout()
            fig.savefig(base_directory + '/plots/' + 'excess_neurons_' + filename + '.pdf')  
            
def CompareSingleNeuronFilteredImages(
    base_directory,
    image_directory,
    disccofanSOM_directory,
    n_neurons,
    m_neurons,
    nx,
    ny,
    n_connectivity = 8,
    n_attributes = 12,
    image_extension = 'jpg',
):
    CleanupTemp(disccofanSOM_directory)
    filenames = sorted(Path( image_directory + '/' ).glob("*."+image_extension))
    with joblib_progress("Filtering neuron ("+ str(nx) + ',' + str(ny) +')' ):
        element_run = Parallel(n_jobs=-1)(delayed(FilterImageFromNeuron)(
            filename = filename.stem,
            base_directory = base_directory,
            disccofanSOM_directory = disccofanSOM_directory,
            n_neurons = n_neurons,
            m_neurons = m_neurons,
            nx = nx, 
            ny = ny,
            n_connectivity = n_connectivity,
            n_attributes = n_attributes,
            image_extension = image_extension) for filename in filenames )    
    print('Filtering completed!')
    
    columns = round(np.sqrt(len(filenames)))
    rows = round(np.sqrt(len(filenames)))

    fig, axs = plt.subplots(rows, columns, figsize=(4.4 * columns, 3.6 * rows) )
    fig.subplots_adjust(left=0.08, bottom=0.13, right=0.95, top=0.9, wspace=0.26, hspace=0.1)
    axs = axs.ravel()

    for ii, ax in enumerate(axs):
        img_i = base_directory + '/filtered/' + filenames[ii].stem + '/' + str(nx) + ',' + str(ny) +'.'+image_extension
        if Path(img_i).is_file:
            ax.imshow(mpimg.imread(img_i), cmap='Greys_r') #cv2.imread(img_i))
        ax.axis('off')
        ax.set_title(filenames[ii].stem, fontsize=14)
        plt.suptitle('Neuron (' + str(nx) + ',' + str(ny) +')', fontsize=17)
    os.makedirs(base_directory + '/plots/', exist_ok=True)
    fig.tight_layout()
    fig.savefig(base_directory + '/plots/' + 'comparison_(' + str(nx) + ',' + str(ny) +')' + '.pdf')  
    
    
def FilterAllNeuronsInOneImage(
    base_directory,
    disccofanSOM_directory,
    filename,
    n_neurons,
    m_neurons,
    n_connectivity = 8,
    n_attributes = 12,
    image_extension = 'jpg'
):
    CleanupTemp(disccofanSOM_directory)
    for nx in range(0,n_neurons):
        with joblib_progress("Filtering "+ filename):
            element_run = Parallel(n_jobs=-1)(delayed(FilterImageFromNeuron)(
                filename,
                base_directory,
                disccofanSOM_directory,
                n_neurons,
                m_neurons,
                nx = nx, 
                ny = ny,
                n_connectivity = n_connectivity,
                n_attributes = n_attributes,
                image_extension = image_extension) for ny in range(0,m_neurons) )    
    print('Filtering completed!')
    
    
def PlotAllNeuronsInOneImage(
    base_directory,
    filename,
    n_neurons,
    m_neurons,
    image_extension = 'jpg'
):
    columns = n_neurons
    rows = m_neurons

    fig, axs = plt.subplots(rows, columns, figsize=(2.04 * columns, 1.6 * rows) )
    fig.subplots_adjust(hspace = 0, wspace = 0, top=0.92, bottom=0.05, left=0.03, right=0.95)

    for nx in range(0,n_neurons):
        for ny in range(0,m_neurons):        
            img_i = base_directory + '/filtered/' + filename + '/' + str(nx) + ',' + str(ny) +'.'+image_extension
            if Path(img_i).is_file():
                axs[m_neurons-ny-1,nx].imshow(mpimg.imread(img_i), cmap='Greys_r') #cv2.imread(img_i))
            axs[m_neurons-ny-1,nx].axis('off')
            axs[m_neurons-ny-1,nx].set_title('(' + Path(img_i).stem + ')', fontsize=14, color='white',y=0.7)
    plt.suptitle(filename, fontsize = 20)
    os.makedirs(base_directory + '/plots/', exist_ok=True)
    fig.tight_layout()
    plt.savefig(base_directory + '/plots/' + 'Filtered_SOM_' + filename + '.pdf')
