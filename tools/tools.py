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
