## Steps in MC weight surgery

1. Find centroids of the 2 classes to be merged 

2. Calculate d, the difference between normalized centroids found above 

3. Calculate Unitary matrix U: 
    - make normalized d the first basis element in the matrix 
    - get 2 other basis vectors 
    - use Gram-Schmidt algorithm to get orthogonal basis vectors 

4. Calculate diagonal matrix S, the orthogonal projection of the first dimension 

5. Calculate inverse of the unitary matrix U, which reverts back to the original basis 