import numpy as np
import matplotlib.pyplot as plt
import os


class Visualize(object):
    def __init__(self, gt, output, neighbor, savedir, epoch):
        # data : [batch, len_total]
        self.gt = gt
        self.output = output
        self.nei = neighbor
        self.savedir = savedir
        self.epoch = epoch
        self.length = 21


    def square(self, img):
        index = []
        # img : [len_total]
        len_total = len(img)
        length = np.int(np.sqrt(len_total))
        base = np.zeros([length, length])
        # Should be odd num
        center = length//2
        # Clockwise direction
        base[center, center] = img[0]
        index.append([center, center])
        cur = 1
        for l in range(1, center+1):
            br = center - l
            bc = center - 1
            base[br, bc] = img[cur]
            index.append([br, bc])
            cur += 1
            # V
            for r in range(l-1):
                br = br
                bc = bc - 1
                base[br, bc] = img[cur]
                index.append([br, bc])
                cur += 1
            # <
            for c in range(2*l):
                br = br + 1
                bc = bc
                base[br, bc] = img[cur]
                index.append([br, bc])
                cur += 1
            # ^
            for r in range(2*l):
                br = br
                bc = bc + 1
                base[br, bc] = img[cur]
                index.append([br, bc])
                cur += 1
            # >
            for c in range(2*l):
                br = br - 1
                bc = bc
                base[br, bc] = img[cur]
                index.append([br, bc])
                cur += 1
            # V
            for r in range(l):
                br = br
                bc = bc - 1
                base[br, bc] = img[cur]
                index.append([br, bc])
                cur += 1

        return base, index

    def region(self, reg, idx):
        base = np.zeros([self.length, self.length])
        for i in range(len(reg)):
            compo = self.nei[i]
            for l in compo:
                base[idx[l][0], idx[l][1]] += reg[i]
        return base




    def plot(self):
        plt.figure(1)

        for dt in range(5):
            cur_gt = self.gt[dt]
            cur_output = self.output[dt]
            a = plt.subplot(2, 5, dt + 1)
            gt, index = self.square(cur_gt)
            a.matshow(gt, cmap='gray')
            b = plt.subplot(2, 5, dt + 5 + 1)
            b.matshow(self.region(cur_output, index), cmap='gray')

        test_dir = os.path.join(self.savedir)
        test_fig = "train%04d" % (self.epoch + 1)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        plt.figure(1).savefig(os.path.join(test_dir, test_fig))
