import numpy as np
import os
import scipy.io as sio
from math import inf
from sklearn.neighbors import NearestNeighbors


## radius = 110
## single size = 10x10

xyz_mat = sio.loadmat(os.getcwd()+"/PASCAL_VOC_2012/cam_xyz_441.mat")
xyz = np.array(xyz_mat['cam_xyz_mat'])



nbrs = NearestNeighbors(n_neighbors=16).fit(xyz)
_, NN = nbrs.kneighbors(xyz)

# NN = np.sort(indices, axis=1)
np.save(os.getcwd()+"/PASCAL_VOC_2012/NN16_441_knn.npy", NN)

print(NN)
