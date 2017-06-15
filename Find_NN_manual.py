import numpy as np
import os
import scipy.io as sio
from math import inf
from sklearn.neighbors import NearestNeighbors


## radius = 110
## single size = 10x10

xyz_mat = sio.loadmat(os.getcwd()+"/PASCAL_VOC_2012/cam_xyz_441.mat")
xyz = np.array(xyz_mat['cam_xyz_mat'])

def cart2sph(xyz):
    x=xyz[0], y=xyz[1], z=xyz[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    th = np.arctan(np.sqrt(x**2 + y**2)/z)
    phi = np.arctan(y/x)
    return [r, th, phi]

def sph2cart(sph):
    r = sph[0], th = sph[1], phi = sph[2]
    xy = r*np.sin(th)
    x = xy*np.cos(phi)
    y = xy*np.sin(phi)
    z = r*np.cos(th)
    return [x, y, z]

def dist3(a, b):
    # distx = a[0] - b[0]
    # disty = a[1] - b[1]
    # distz = a[2] - b[2]
    return np.linalg.norm(a-b)

level_list = []
cur_level = []
level = []
last = []
level_s = [0]
l = 0
for i in range(len(xyz)):
    if xyz[i][2] != xyz[i-1][2] and i!=0:
        level_list.append(cur_level)
        cur_level = []
        level_s.append(i)
        l += 1
        last[-1] = True
    last.append(False)
    level.append(l)
    cur_level.append(xyz[i])
    # print(xyz[i], level[i])
level_list.append(cur_level)
level_s.append(len(xyz))
# print(level_s)
l_max = l
neighbor = [[1, 2, 0, 8, 3, 4, 5, 6, 7]]


idx = 1
for j in range(1, len(level_s)-2):
    for k in range(level_s[j+1]-level_s[j]):
        upfloor = NearestNeighbors(n_neighbors=1).fit(level_list[j-1])
        myfloor = NearestNeighbors(n_neighbors=3).fit(level_list[j])
        downfloor = NearestNeighbors(n_neighbors=5).fit(level_list[j + 1])
        _, up = upfloor.kneighbors(xyz[idx])
        up = [level_s[j-1]+up[0,0]]
        _, my = myfloor.kneighbors(xyz[idx])
        my = np.sort([level_s[j]+my[0,0],level_s[j]+my[0,1], level_s[j]+my[0,2]])
        while(my[-1]-my[0] > 2 or my[-1]-my[0] == -1):
            tail = my[-1]
            my[2] = my[1]
            my[1] = my[0]
            my[0] = tail

        _, down = downfloor.kneighbors(xyz[idx])
        down = np.sort([level_s[j+1]+down[0,0], level_s[j+1]+down[0,1], level_s[j+1]+down[0,2],
                level_s[j+1]+down[0,3], level_s[j+1]+down[0,4]])
        while (down[-1] - down[0] > 4 or down[-1] - down[0] == -1):
            tail = down[-1]
            down[4] = down[3]
            down[3] = down[2]
            down[2] = down[1]
            down[1] = down[0]
            down[0] = tail

        cur_neighbor = np.concatenate((up,my,down))
        neighbor.append(cur_neighbor)
        idx += 1

np.save(os.getcwd() + "/PASCAL_VOC_2012/NN_441_manual.npy", neighbor)
        # print(down)




