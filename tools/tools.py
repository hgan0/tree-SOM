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

def LoadTrees():

def Train():








###### load all data sets

file_names = []
data_dir= '/home/hgan/notebooks/delaysp_hpf/trees/'
path_of_the_directory = data_dir
ext = ('tree.csv')
data = pd.DataFrame()

for path, dirc, files in os.walk(path_of_the_directory):
    sampled_files = random.choices(files, k=round(len(id_set)*0.1))
    for name in sampled_files:
        if name.endswith(ext):
            file_names.append(name)
            #print(name[:-4])
            print(name)
#             try:                
            tmp = pd.read_csv(data_dir+name, sep=',', header=0, low_memory=False) 
            #tmp = tmp.assign(LST=lambda x: tmp.iloc[:,-1] + int(find_between( name, '_t00', '-I-')))
            data = pd.concat([data, tmp], ignore_index=True)
            #data = data.append(tmp, ignore_index=True)
            print('appending..')
