import scipy.io as sio
import numpy as np
import os
import h5py

eye_size = 10
# rot = 0
name = "img"


# Load Data from .mat files
cwd = os.getcwd()
compoundData = h5py.File(cwd+"/coco/%d/%s.mat" %(eye_size, name) )
compoundData.keys()
# compoundData_seg = h5py.File(cwd+"/coco/%d/seg_%s.mat" %(eye_size, name))
# compoundData_seg.keys()

# Save directories
datadir = cwd+'/coco/%d/test/%s_pre' %(eye_size, name)
# knndir = cwd+'/coco/%d/rint(randint(-3, 3))seg_%s_knn' % (eye_size, name)
# tridir = cwd+'/coco/%d/seg_%s_tri' % (eye_size, name)
# gtdir = cwd+'/coco/%d/gt_%s' % (eye_size, name)
#
if not os.path.exists(datadir):
    os.makedirs(datadir)
# if not os.path.exists(knndir):
#     os.makedirs(knndir)
# if not os.path.exists(tridir):
#     os.makedirs(tridir)
# if not os.path.exists(gtdir):
#     os.makedirs(gtdir)

#
# transpose data to [batch, eyes, width, height, channel], [batch, 441, 10, 10, 3]
compoundData = compoundData['compoundData']
compoundData = np.transpose(compoundData, (0, 1, 3, 4, 2))
compoundData = compoundData[9000:]
compound_mean = np.mean(compoundData, axis=0)
# compoundData_seg = compoundData_seg['compoundData']
# compoundData_seg = np.transpose(compoundData_seg, (0, 1, 3, 4, 2))
# compoundData_rs = np.reshape(compoundData_seg, (np.shape(compoundData_seg)[0],np.shape(compoundData_seg)[1], -1))
# compoundData_gt = np.mean(compoundData_rs, axis=2)
# gt_single = np.copy(compoundData_gt)

# # knn
# knn = np.load(os.getcwd()+"/NN/NN_441_knn.npy")
# shape = np.shape(compoundData_gt)
# knn_val = np.zeros([shape[0], np.shape(knn)[0]])
# for reg in range(np.shape(knn)[0]):
#     knn_val[:, reg] = np.mean(gt_single[:, knn[reg]], axis=1)

# Tri
# tri = np.load(os.getcwd()+"/NN/NN_441_tri.npy")
# shape = np.shape(compoundData_gt)
# tri_val = np.zeros([shape[0], np.shape(tri)[0]])
# for reg in range(np.shape(tri)[0]):
#     tri_val[:, reg] = np.mean(gt_single[:, tri[reg]], axis=1)

# np.save(datadir+'/mean', compound_mean)
mean = np.load(cwd+'/coco/%d/%s_pre/mean.npy' %(eye_size, name))

for i in range(len(compoundData)):
    np.save(datadir+'/%05d' % (i+9000), compoundData[i]-compound_mean)
    # np.save(knndir+'/%05d' % (i), knn_val[i])
    # np.save(tridir+'/%05d' % (i), tri_val[i])
    # np.save(gtdir+'/%05d' % (i), compoundData_gt[i])
    print("Process %s, %dth image Finished" %(name, i+1))