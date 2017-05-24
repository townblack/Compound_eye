import os
import numpy as np

cwd = os.getcwd()

original = np.load(cwd+'/0.2/compoundData_train.npy')
original_seg = np.load(cwd+'/0.2/compoundData_seg_train.npy')

aug1 = np.load(cwd+'/0.2/compoundData_train_color.npy')
aug1_seg = np.copy(original_seg)
aug1_seg2 = np.load(cwd+'/0.2/compoundData_seg_test.npy')


concat = np.concatenate((original, aug1), axis=0)
concat_seg = np.concatenate((original_seg, aug1_seg, aug1_seg2), axis=0)

np.save(cwd+'/0.2/compoundData_train_aug.npy', concat)
np.save(cwd+'/0.2/compoundData_seg_train_aug.npy', concat_seg)