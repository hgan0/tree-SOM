from pathlib import Path
from tqdm.notebook import tqdm
from astropy.io import fits
from minisom import MiniSom  

import os
import pandas as pd
import minisom 
import pickle

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
    directory: str, 
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
    imagefiles = sorted(Path(directory).glob("*."+extension))
    return imagefiles
    
def CreateFits_for_CHIME(
    ids: array, 
    save_directory: str,
    revision = "rev_07",
):
    for ii, csd in tqdm(enumerate(ids)):
    csd = str(csd)
    daily_processing_path = Path(f"/project/rpp-chime/chime/chime_processed/daily/{revision}/")

    try:
        name = "delayspectrum_lsd_" + csd + ".h5"
        ds = containers.DelaySpectrum.from_file( daily_processing_path / csd / name)
        newimage = lscale(ds.spectrum[:].T)
        os.makedirs(save_directory, exist_ok=True)
        astropy.io.fits.writeto(save_directory+'/'+csd+'.fits', newimage, header=None,)            
    except: 
        print("Failed to create a fit file!")

def CreateTrees(
    imagefiles: list,
    extension: str,
    disccofan_directory: str,
    save_directory: str,
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
        outname = save_directory + '/' + filename + str("_tree")
        command = '%s -g 1,1,1 --threads %d  -v info -c %d  -a %d \
        --inprefix %s --intype %s -o treeCSV --outprefix %s -l %d'%(disccofan_directory, n_thread, c, a, filename, intype, outname, lval)
        
        try: 
            os.system(command)
        except:
            print("Failed to create a tree..!")
                           
def LoadTrees(
    tree_directory : str,
    extension = "csv", 
    sampling_rate = 1.0,     
):
    treefiles = sorted(Path(tree_directory).glob("*."+extension))
    data = pd.DataFrame()    
    sampled_files = random.choices(treefiles, k=round(len(treefiles)*sampling_rate))
    
    for tree_file in tqdm(sampled_files):
        tmp = pd.read_csv(tree_file, sep=',', header=0, low_memory=False) 
        data = pd.concat([data, tmp], ignore_index=True)
    return trim_data(data)


def TrainSOM(
    train_set,
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
	
	with open(base_directory+'som-'+training_no+'.p', 'wb') as outfile:
    		pickle.dump(som, outfile)

def GetWinningNeurons(
	
):

def GetNormalisedExcess(
	
):

def GetPatternSpectrum(
	
):
