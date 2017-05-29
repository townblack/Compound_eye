import scipy.io as sio
import numpy as np
import os
import h5py

# Load Data from .mat files
cwd = os.getcwd()
compoundData = h5py.File(cwd+"/PASCAL_VOC_2012/compoundData_test.mat")
compoundData.keys()
compoundData_seg = h5py.File(cwd+"/PASCAL_VOC_2012/compoundData_seg_test.mat")
compoundData_seg.keys()

# compoundData = sio.loadmat(cwd+"/PASCAL_VOC_2012/compoundData_train_color.mat")
# compoundData_seg = sio.loadmat(cwd+"/PASCAL_VOC_2012/compoundData_seg_train_flip.mat")
# print(compoundData)

# transpose data to [batch, width, height, channel], [422, 21, 21, 300]
compoundData = compoundData['compoundData']
compoundData = np.transpose(compoundData,(0,2,3,1))
# compoundData = np.transpose(compoundData,(3,0,1,2))
compoundData_seg = compoundData_seg['compoundData2']
compoundData_seg = np.transpose(compoundData_seg,(0,2,3,1))
# compoundData_seg = np.transpose(compoundData_seg,(3,0,1,2))

# Compute the mean of object ground truth by each channel
# if the mean value if over 0.2 --> that pixel is true (have object)
# [batch, width, height, channel, 1]
compoundData_seg_mean = np.mean(compoundData_seg, axis=3)
compoundData_seg_01 = [np.copy(compoundData_seg_mean>0.2)]
compoundData_seg_01 = np.transpose(compoundData_seg_01, (1,2,3,0))

compoundData = np.flip(compoundData, 2)
compoundData_seg_01 = np.flip(compoundData_seg_01, 2)



np.save(cwd+'/PASCAL_VOC_2012/0.2/compoundData_test.npy', compoundData)
np.save(cwd+'/PASCAL_VOC_2012/0.2/compoundData_seg_test.npy', compoundData_seg_01)

print("Process Finished")