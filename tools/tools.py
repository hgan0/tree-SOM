from pathlib import Path
from tqdm.notebook import tqdm
from astropy.io import fits
from minisom import MiniSom
import numpy as np
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
    ids: array, 
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
    imagefiles: list,
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
    for file in tqdm(imagefiles):
        filename = file.stem
        outname = base_directory + '/trees/' + filename + str("_tree")
        command = '%s -g 1,1,1 --threads %d  -v info -c %d  -a %d \
        --inprefix %s --intype %s -o treeCSV --outprefix %s -l %d'%(disccofan_directory, n_thread, c, a, filename, intype, outname, lval)
        
        try: 
            os.system(command)
        except:
            print("Failed to create a tree..!")
                           
def LoadTrees(
    base_directory : str,
    extension = "csv", 
    sampling_rate = 1.0,     
):
    treefiles = sorted(Path(base_directory+'/trees/').glob("*."+extension))
    data = pd.DataFrame()    
    sampled_files = random.choices(treefiles, k=round(len(treefiles)*sampling_rate))
    
    for tree_file in tqdm(sampled_files):
        tmp = pd.read_csv(tree_file, sep=',', header=0, low_memory=False) 
        data = pd.concat([data, tmp], ignore_index=True)
    return trim_data(data)


def TrainSOM(
    train_set,
    base_directory,
    s_features = [4,5,6,8,9,13,14],
    n_neurons = 10,
    m_neurons = 10,
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
    s_features : 
    
    """
	flat_data = []
	features = np.array(list(train_set.keys()))
	
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
	plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
	plt.colorbar()
	plt.savefig(base_directory, 'SOM_distance_map.pdf')
	plt.clf()
	with open(base_directory+'som.p', 'wb') as outfile:
    		pickle.dump(som, outfile)

def GetWinningNeurons(
	base_directory,
	map_size,
	tree_extension = 'csv'
):
	file_names = sorted(Path(tree_directory).glob("*."+extension))
	
	for file in tqdm(file_names):
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
	
	    map_size=str(n_neurons)+'x'+str(m_neurons)
	
	    df2 =  pd.DataFrame({'index': flat_data[0],
	           'winning_Nx': w_x,
	           'winning_Ny': w_y,
	                        })
	    os.makedirs(base_directory+'/SOMs/', exist_ok=True)  
	    df2 = df2.loc[:, ~df2.columns.str.contains('^Unnamed')]
	    df2 = df2.astype(int)
	    df2.to_csv(base_directory+'/SOMs/'+filename+'_'+map_size+'SOMs.csv')  

def CreateGValueFluxTables(
	base_directory: str,
	som_size: int,
):
	id_set = sorted(all_set)
	tree_directory = base_directory + '/trees/'
	par1 = 'gval'
	par2 = 'flux'
	sum_table = pd.DataFrame({
	                'csd': [],
	                'winning_Nx': [],
	                'winning_Ny': [],
	                'sum_'+par1:[], 
	                'sum_'+par2:[]}, index=[])
	ext = ('.csv')
	a = 0
	for obs in tqdm(id_set):
	    attributes = pd.read_csv(base_directory 
	                             + 'trees/', sep=',', header=0, low_memory=False) 
	    winning_neurons = pd.read_csv(base_directory 
	                                  + 'SOMs/SOMs.csv', sep=',', header=0, low_memory=False) 	
	    flux = pd.DataFrame({
	                    'winning_Nx': [],
	                    'winning_Ny': [],
	                    'sum_par1':[],
	                    'sum_par2':[]}, index=[])
	    img_par1 = np.zeros(som_size*som_size).reshape((som_size, som_size))
	    img_par2 = np.zeros(som_size*som_size).reshape((som_size, som_size))
	
	    for nx in range(0,som_size):
	        for ny in range(0,som_size):
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
	            
	            img_par1[som_size-1-nx,ny] = sum_par1
	            img_par2[som_size-1-nx,ny] = sum_par2
	
	            tmp =  pd.DataFrame({
	                'id': id,
	                'winning_Nx': nx,
	                'winning_Ny': ny,
	                'sum_'+par1: sum_par1,  
	                'sum_'+par2: sum_par2}, index=[a])
	            a = a + 1
	            sum_table = pd.concat([sum_table,tmp], ignore_index=True)
	            
	    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.))
	
	    im1 = ax[0].imshow((img_par1), extent=[0,som_size,som_size,0])
	    ax[0].set_xlabel('Neuron_x')
	    ax[0].set_ylabel('Neuron_y')
	    ax[0].set_title(par1)
	    cbar1 = plt.colorbar(im1, ax=ax[0])
	    cbar1.set_label(par1)
	    cbar1.formatter.set_powerlimits((0, 0))
	
	    im2 = ax[1].imshow((img_par2), extent=[0,som_size,0,som_size])
	    cbar2 = plt.colorbar(im2, ax=ax[1])
	    cbar2.set_label(par2)
	    cbar2.formatter.set_powerlimits((0, 0))
	    ax[1].set_xlabel('Neuron_x')
	    ax[1].set_ylabel('Neuron_y')
	    ax[1].set_title(par2)
		
	    os.makedirs(base_directory + 'sum_tables/', exist_ok=True)
	    fig.suptitle('csd'+str(obs))
	    fig.subplots_adjust(left=0.08, bottom=0.13, right=0.95, top=0.9, wspace=0.26, hspace=0.1)
	    fig.savefig(base_directory + 'sum_tables/'+ str(obs)+'_summary.pdf')
	    fig.clf()
	
	os.makedirs(base_directory + 'sum_tables/', exist_ok=True)        
	sum_table.to_csv(base_directory + 'sum_tables/sum_table.csv')

def GetNormalisedExcess(
	
):

def GetPatternSpectrum(
	
):
