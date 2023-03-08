# weight_surgery
implementing weight surgery, visualizing embeddings

## Setting up the environment

Create anaconda workspace
```
conda create -n weight_surgery python=3.7 -y 
```

Activate workspace and install dependencies
```
conda activate weight_surgery

pip install -r requirements.txt
```


## Steps in weight surgery

1. Find centroids of the 2 classes to be merged 

2. Calculate d, the difference between normalized centroids found above 

3. Calculate Unitary matrix U: 
    - make normalized d the first basis element in the matrix 
    - get 2 other basis vectors 
    - use Gram-Schmidt algorithm to get orthogonal basis vectors 

4. Calculate diagonal matrix S, the orthogonal projection of the first dimension 

5. Calculate inverse of the unitary matrix U, which reverts back to the original basis 


## Running the main notebook 

Run notebook to perform weight surgery and visualize embeddings in 3D plots
```
jupyter notebook Weight_Surgery_MNIST.ipynb
```
