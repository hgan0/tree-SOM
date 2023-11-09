## Self-organising attribute maps and pattern spectra

This project combines Max tree data structure (implemented by [DISCCOFAN](https://github.com/sgazagnes/disccofan) and [DISCCOFAN-SOM](https://github.com/sgazagnes/disccofan/tree/SOM)) and Self Organising Maps (implemented by [minisom](https://github.com/JustGlowing/minisom/tree/master)) to develop self-organising attribute maps and pattern spectra.
This tool explores morphological structures in images. 

To use the treeSOM code, you need to install DISCCOFAN (disccofan-master and disccofan-SOM for filtering based on the SOM) and minisom first, following the instructions in the links above. 

## How to use treeSOM?

First, create a work directory to store your data (images) and results. 
Your training images should be stored in `image_directory`.
`tree_directory` corresponds to the directory where max trees of the training images will be stored. 
(You can also directly put your tree files into `tree_directory`)

```python
base_directory = '/project/treeSOM/'
image_directory = base_directory + '/images/'
tree_directory = base_directory + '/trees/'
disccofan_directory = '/project/software/disccofan-master/'
disccofanSOM_directory = '/project/software/disccofan-SOM/'
     
```

Load image files to train. 

```python
imagefiles = TreeSOM.LoadImageDirectory(image_directory, extension='fits')
     
```

Plot training images all at once.

```python
TreeSOM.ShowImages_in_Parallel(
    imagefiles,
    base_directory,
    cmap = 'inferno',
)
     
```

Use `ShowImages_in_Parallel` instead, if the number of training images is larger than 25. 

Create max trees for the training images. 

```python
TreeSOM.CreateTrees_in_Parallel(imagefiles,
    'fits',
    disccofan_directory,
    base_directory,
    tree_directory, )
     
```

Load (sampled) vector attributes from nodes of the max trees. 
`tree_selecting_rate` indicates the fraction of trees to load from the entire data sets and `sampling_rate` indicates the fraction of nodes to load within each tree. 

```python
train_set = TreeSOM.LoadTrees(
    base_directory,
    tree_directory,
    extension = "csv", 
    tree_selecting_rate = 0.1,
    sampling_rate = 0.005,   )

```

Set parameters for the self-organising map training and train a SOM. 

```python
n_neurons = 15
m_neurons = 15
s_features = [4,5,6,8,9,13,14]
max_iter = 100000

TreeSOM.TrainSOM(train_set, base_directory, n_neurons, m_neurons, s_features, max_iter,)

```
