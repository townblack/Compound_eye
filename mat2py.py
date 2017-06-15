import scipy.io as sio
import numpy as np
import os
import h5py

# Load Data from .mat files
cwd = os.getcwd()
# compoundData = h5py.File(cwd+"/PASCAL_VOC_2012/compoundData_test.mat")
# compoundData.keys()
compoundData_seg = h5py.File(cwd+"/PASCAL_VOC_2012/compoundData_seg_test.mat")
compoundData_seg.keys()

# compoundData = sio.loadmat(cwd+"/PASCAL_VOC_2012/compoundData_train_color.mat")
# compoundData_seg = sio.loadmat(cwd+"/PASCAL_VOC_2012/compoundData_seg_train_flip.mat")
# print(compoundData)


#
# transpose data to [batch, eyes, width, height, channel], [batch, 441, 10, 10, 3]
# compoundData = compoundData['compoundData']
# compoundData = np.transpose(compoundData, (0, 1, 3, 4, 2))
compoundData_seg = compoundData_seg['compoundData']
compoundData_seg = np.transpose(compoundData_seg, (0, 1, 3, 4, 2))
compoundData_rs = np.reshape(compoundData_seg, (np.shape(compoundData_seg)[0],np.shape(compoundData_seg)[1], -1))
compoundData_gt = np.mean(compoundData_rs, axis=2)
gt_single = np.copy(compoundData_gt)
neighbor = np.load(os.getcwd()+"/PASCAL_VOC_2012/NN_441_manual.npy")
# for eye in range(np.shape(compoundData_gt)[1]):
#     region_val = np.mean(gt_single[:, neighbor[eye]], axis=1)
#     compoundData_gt[:,eye] = region_val
shape = np.shape(compoundData_gt)
region_val = np.zeros([shape[0], np.shape(neighbor)[0]])
for reg in range(np.shape(neighbor)[0]):
    region_val[:, reg] = np.mean(gt_single[:, neighbor[reg]], axis=1)

# Classification
# shape = np.shape(compoundData_gt)
# comp_class = np.zeros([shape[0], shape[1], 2])
# comp_class[:,:,1] = np.copy(compoundData_gt)
# comp_class[:,:,0] = np.copy(1-compoundData_gt)

# np.save(cwd+'/PASCAL_VOC_2012/compoundData_test.npy', compoundData)
np.save(cwd+'/PASCAL_VOC_2012/compoundData_seg_test_manual.npy', region_val)

print("Process Finished")