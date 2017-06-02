import numpy as np
import os
import scipy.io as sio
from math import inf
from sklearn.neighbors import NearestNeighbors


## radius = 110
## single size = 10x10

xyz_mat = sio.loadmat(os.getcwd()+"/VOC2012/VOC2012_comp/cam_xyz_441.mat")
xyz = np.array(xyz_mat['cam_xyz_mat'])



nbrs = NearestNeighbors(n_neighbors=9).fit(xyz)
_, indices = nbrs.kneighbors(xyz)

NN = np.sort(indices, axis=1)
np.save(os.getcwd()+"/VOC2012/VOC2012_comp/NN_441.npy", NN)

print(NN)
