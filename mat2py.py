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
compoundData_gt = [np.copy(compoundData_gt>0.5)]
# compoundData_gt_2 = [np.copy(compoundData_gt>0.7)]
compoundData_gt = np.transpose(compoundData_gt, (1, 2, 0))

# np.save(cwd+'/PASCAL_VOC_2012/compoundData_test.npy', compoundData)
np.save(cwd+'/PASCAL_VOC_2012/compoundData_seg_test.npy', compoundData_gt)

print("Process Finished")