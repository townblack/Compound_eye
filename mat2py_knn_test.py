import scipy.io as sio
import numpy as np
import os
import h5py

eye_size = 10
# rot = 0
name = "test"

for k in [1,3,5,7,9,11,13,15]:
# Load Data from .mat files
    cwd = os.getcwd()
    compoundData_seg = h5py.File(cwd+"/PASCAL_VOC_2012/%d/seg_%s.mat" % (eye_size, name))
    compoundData_seg.keys()

    # Save directories
    knndir = cwd+'/PASCAL_VOC_2012/%d/knn_test/%d' % (eye_size, k)
    if not os.path.exists(knndir):
        os.makedirs(knndir)

    #
    # transpose data to [batch, eyes, width, height, channel], [batch, 441, 10, 10, 3]
    compoundData_seg = compoundData_seg['compoundData']
    compoundData_seg = np.transpose(compoundData_seg, (0, 1, 3, 4, 2))
    compoundData_rs = np.reshape(compoundData_seg, (np.shape(compoundData_seg)[0],np.shape(compoundData_seg)[1], -1))
    compoundData_gt = np.mean(compoundData_rs, axis=2)
    gt_single = np.copy(compoundData_gt)

    # knn
    knn = np.load(os.getcwd()+"/PASCAL_VOC_2012/NN/NN%d_441_knn.npy" %(k))
    shape = np.shape(compoundData_gt)
    knn_val = np.zeros([shape[0], np.shape(knn)[0]])
    for reg in range(np.shape(knn)[0]):
        knn_val[:, reg] = np.mean(gt_single[:, knn[reg]], axis=1)

    for i in range(len(compoundData_gt)):
        np.save(knndir+'/%05d' % (i), knn_val[i])

    print("Process  NN %d, %s Finished" %(k, name))