from pathlib import Path

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
    
#def CreateFits_for_CHIME():

def CreateTrees(
    imagefiles: list,
    extension: str,
    DISCCOFAN_directory: str,
    save_directory: str,
    c = 8,
    a = 12,
    lval = 100,
    
):
    for file in imagefiles:
        filename = file.stem
        outname = save_directory + '/' + filename + str("_tree")
        command = '%s -g 1,1,1 --threads 1  -v info -c %d  -a %d \
        --inprefix %s --intype %s -o treeCSV --outprefix %s -l %d'%(DISCCOFAN_directory, c, a, filename, intype, outname, lval)
        
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
    ext = ('.'+extension)
    data = pd.DataFrame()    

    for path, dirc, files in os.walk(tree_directory):
        sampled_files = random.choices(files, k=round(len(id_set)*0.1))
        for name in sampled_files:
            if name.endswith(ext):
                file_names.append(name) 
                tmp = pd.read_csv(data_dir+name, sep=',', header=0, low_memory=False) 
                data = pd.concat([data, tmp], ignore_index=True)
                
def Train():







