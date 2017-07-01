import numpy as np
import os
from random import randint

# a = False



datadir = os.getcwd()+'/PASCAL_VOC_2012/10/test_pre'
a = np.load(datadir + '/00001.npy')

datadir2 = os.getcwd()+'/PASCAL_VOC_2012/10/test'
b = np.load(datadir2 + '/00001.npy')

print(randint(-3, 3))