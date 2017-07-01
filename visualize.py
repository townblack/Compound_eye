import numpy as np
import matplotlib.pyplot as plt
import os


class Visualize(object):
    def __init__(self, gt, label, output, neighbor, savedir, epoch):
        # data : [batch, len_total]
        self.gt = gt
        self.label = label
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
        count = np.zeros([self.length, self.length])
        for i in range(len(reg)):
            compo = self.nei[i]
            for l in compo:
                base[idx[l][0], idx[l][1]] = count[idx[l][0], idx[l][1]] * base[idx[l][0], idx[l][1]] + reg[i]
                count[idx[l][0], idx[l][1]] += 1
                base[idx[l][0], idx[l][1]] = base[idx[l][0], idx[l][1]]/count[idx[l][0], idx[l][1]]

        classify = np.zeros([self.length, self.length])
        for i in range(len(reg)):
            compo = self.nei[i]
            if reg[i] > 0.5:
                for l in compo:
                    classify[idx[l][0], idx[l][1]] = 1



        return base, classify

    def bincl(self, target):
        binary = np.copy(target)
        for i in range(np.shape(target)[0]):
            for j in range(np.shape(target)[1]):
                if target[i][j]>0.5:
                    binary[i][j] = 1
                else:
                    binary[i][j] = 0

        return binary



    def plot(self):
        plt.figure(1)

        for dt in range(5):
            cur_gt = self.gt[dt]
            cur_label =self.label[dt]
            cur_output = self.output[dt]
            a = plt.subplot(4, 5, dt + 1)
            plt.axis('off')
            gt, index = self.square(cur_gt)
            a.matshow(gt, cmap='gray')
            ar = plt.subplot(4, 5, dt +5 + 1)
            plt.axis('off')
            label, _ = self.square(cur_label)
            ar.matshow(label, cmap='gray')
            ditb, clss = self.region(cur_output, index)
            b = plt.subplot(4, 5, dt + 10 + 1)
            plt.axis('off')
            b.matshow(ditb, cmap='gray')
            c = plt.subplot(4, 5, dt + 15 + 1)
            plt.axis('off')
            c.matshow(clss, cmap='gray')

        test_dir = os.path.join(self.savedir)
        test_fig = "train%04d" % (self.epoch + 1)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        plt.figure(1).savefig(os.path.join(test_dir, test_fig))
