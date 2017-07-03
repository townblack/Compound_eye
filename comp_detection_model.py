import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
from visualize import Visualize
from random import randint
import scipy.misc
import os


def lrelu(x, alpha=0.2, max_value=None):
    return tf.maximum(x, alpha*x)

class DET(object):
    def __init__(self, sess, image_size=10, image_dim=3, eye_num= 441, n_number=9):

        self.sess = sess
        self.image_size = image_size
        self.image_dim = image_dim
        self.eye_num = eye_num
        self.neighbor = np.load(os.getcwd()+"/NN/NN%d_441_knn.npy" % (n_number))
        self.reg_num = np.shape(self.neighbor)[0]
        self.nei_num = np.shape(self.neighbor)[1]
        n_idx = []
        for s in range(self.reg_num):
            c_idx = []
            for n in range(self.nei_num):
                c_idx.append([s, self.neighbor[s][n]])
            n_idx.append(c_idx)
        self.n_idx = n_idx
        self.rand_num = 80
        self.counter = tf.Variable(0, trainable=False)
        # self.batch_size = batch_size
        self.build_model()


    def build_model(self):
        self.input = tf.placeholder(tf.float32, [None, self.eye_num, self.image_size, self.image_size, self.image_dim])
        # classification
        # self.gt = tf.placeholder(tf.float32, [None, self.eye_num, 2])
        # regression
        self.gt = tf.placeholder(tf.float32, [None, self.reg_num])
        self.batch_size = tf.shape(self.input)[0]
        self.is_training = tf.placeholder(tf.bool)

        self.enc = self.encoder(self.input, self.is_training)
        self.det, self.randsel = self.detector(self.enc, self.is_training)
        self.det_test = self.detector_test(self.enc, self.is_training)


        # L1 Loss
        # self.loss = tf.reduce_mean(tf.abs(self.gt - self.det))
        sel_gt = tf.gather(tf.transpose(self.gt, [1, 0]), self.randsel)
        sel_gt = tf.transpose(sel_gt, [1, 0])
        self.gt_full = tf.reshape(sel_gt, [-1])
        self.det_full = tf.reshape(self.det, [-1])
        self.gt_full_test = tf.reshape(self.gt, [-1])
        self.det_full_test = tf.reshape(self.det_test, [-1])
        with tf.name_scope("loss") as scope:
            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt_full, logits=det_full))
            self.loss = tf.divide(tf.nn.l2_loss(self.gt_full - self.det_full), tf.to_float(tf.shape(self.det_full)[0]))
            self.loss_test = tf.divide(tf.nn.l2_loss(self.gt_full_test - self.det_full_test),
                                                        tf.to_float(tf.shape(self.det_full_test)[0]))
            # self.loss = tf.reduce_mean(tf.abs(self.gt_full - self.det_full))
            # self.loss_test = tf.reduce_mean(tf.abs(self.gt_full_test - self.det_full_test))

            ########################
            # New loss mx

            # self.loss = tf.divide(tf.nn.l2_loss(self.gt_full - self.det_full) \
            #             + tf.reduce_sum(tf.maximum(0., -(self.gt_full-0.5)*(self.det_full-0.5)))
            #             , tf.to_float(tf.shape(self.gt_full)[0]))
            # self.loss_test = tf.divide(tf.nn.l2_loss(self.gt_full_test - self.det_full_test) \
            #                  + tf.reduce_sum(tf.maximum(0., -(self.gt_full_test-0.5)*(self.det_full_test-0.5)))
            #                 , tf.to_float(tf.shape(self.gt_full_test)[0]))


            ################################
            # Regression + cross entropy
            #################################
            # gt_full_op = 1.-self.gt_full
            # gt_full_2 = tf.concat([tf.reshape(self.gt_full,[-1,1]), tf.reshape(gt_full_op,[-1,1])], axis=1)
            # det_full_op = 1. - self.det_full
            # det_full_2 = tf.concat([tf.reshape(self.det_full,[-1,1]), tf.reshape(det_full_op,[-1,1])], axis=1)
            #
            # gt_full_test_op = 1. - self.gt_full_test
            # gt_full_test_2 = tf.concat([tf.reshape(self.gt_full_test,[-1,1]), tf.reshape(gt_full_test_op,[-1,1])], axis=1)
            # det_full_test_op = 1. - self.det_full_test
            # det_full_test_2 = tf.concat([tf.reshape(self.det_full_test,[-1,1]), tf.reshape(det_full_test_op, [-1,1])], axis=1)
            #
            # self.loss = tf.nn.l2_loss(self.gt_full - tf.nn.sigmoid(self.det_full))\
            #             + tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=gt_full_2, logits=det_full_2))
            # self.loss_test = tf.nn.l2_loss(self.gt_full_test - tf.nn.sigmoid(self.det_full_test)) \
            #                  + tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=gt_full_test_2, logits=det_full_test_2))


            #########################################

            ##########################################
            # Chanho loss
            # self.loss = tf.divide(tf.nn.l2_loss(self.gt_full - self.det_full) \
            #                       + tf.reduce_sum(-(self.gt_full - 0.5) * self.det_full)
            #                       , tf.to_float(tf.shape(self.gt_full)[0]))
            # self.loss_test = tf.divide(tf.nn.l2_loss(self.gt_full_test - self.det_full_test) \
            #                            + tf.reduce_sum(-(self.gt_full_test - 0.5) * self.det_full_test )
            #                            , tf.to_float(tf.shape(self.gt_full_test)[0]))


            #################################################

            self.loss_sum_train = tf.summary.scalar("loss_train", self.loss_test)
            self.loss_sum_test = tf.summary.scalar("loss_test", self.loss_test)
        # self.corr = tf.equal(tf.arg_max(det_full,1), tf.arg_max(gt_full,1))
        self.corr = tf.equal(self.gt_full>0.5, self.det_full>0.5)
        self.corr_test = tf.equal(self.gt_full_test>0.5, self.det_full_test>0.5)
        # with tf.name_scope("accr") assc
        with tf.name_scope("accr") as scope:
            self.accr = tf.reduce_mean(tf.cast(self.corr, "float"))
            self.accr_t = tf.reduce_mean(tf.cast(self.corr_test, "float"))
            self.accr_train = tf.summary.scalar("accr_train", self.accr_t)
            self.accr_test = tf.summary.scalar("accr_test", self.accr_t)
        self.vars = tf.trainable_variables()
        self.saver = tf.train.Saver()


    def encoder(self, _im, is_training=True):
        conv_init_params = tf.truncated_normal_initializer(stddev=0.02)
        fully_init_params = tf.random_normal_initializer(stddev=0.02)
        bias_init_params = tf.constant_initializer(0.0)
        bn_init_params = {'beta': tf.constant_initializer(0.),
                          'gamma': tf.random_normal_initializer(1., 0.02)}
        bn_params = {'is_training': is_training, 'decay': 0.9, 'epsilon': 1e-5,
                     'param_initializers': bn_init_params, 'updates_collections': None}


        # batch, 441 -> full batch
        _cells = tf.reshape(_im, (-1, self.image_size, self.image_size, self.image_dim))

        with tf.variable_scope("encoder") :
            _e0 = slim.conv2d(_cells, 64, [3, 3], stride=2, activation_fn=tf.nn.relu,
                                  weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                  normalizer_fn=slim.batch_norm, normalizer_params=bn_params,
                                  scope='conv0')
            _e1 = slim.conv2d(_e0, 128, [3, 3], stride=2, activation_fn=tf.nn.relu,
                                  weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                  normalizer_fn=slim.batch_norm, normalizer_params=bn_params,
                                  scope='conv1')
            _embed = slim.fully_connected(tf.reshape(_e1, [-1, 3*3*128]), 256, activation_fn=tf.nn.relu,
                                  weights_initializer=fully_init_params, biases_initializer=bias_init_params,
                                  normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='fconv')

        # full batch -> batch, 441

        _embed_comp = tf.reshape(_embed, (-1, self.eye_num, 256))

        return _embed_comp


    def detector(self, _embed, is_training=True):
        conv_init_params = tf.truncated_normal_initializer(stddev=0.02)
        fully_init_params = tf.random_normal_initializer(stddev=0.02)
        bias_init_params = tf.constant_initializer(0.0)
        bn_init_params = {'beta': tf.constant_initializer(0.),
                          'gamma': tf.random_normal_initializer(1., 0.02)}
        bn_params = {'is_training': is_training, 'decay': 0.9, 'epsilon': 1e-5,
                     'param_initializers': bn_init_params, 'updates_collections': None}

        # Choose random place
        randsel = np.random.choice(21*21, self.rand_num, replace=False)
        np.sort(randsel)
        rand_reg = tf.reshape(tf.gather(self.neighbor, randsel), [-1])
        # selneig = np.take(self.n_idx,randsel, axis=0)
        # for o in range(self.rand_num):
        #     for n in range(self.nei_num):
        #         selneig[o,n,0] = o

        with tf.variable_scope("detector") :
            region = tf.transpose(_embed, [1, 0, 2])
            region = tf.transpose(tf.gather(region, rand_reg), [1,0,2])
            region = tf.reshape(region, [-1, self.rand_num, self.nei_num, 256])
            # region = tf.transpose(_embed, [2, 1, 0, 3])
            # region = tf.tile(region, [self.rand_num, 1, 1, 1])
            # region = tf.gather_nd(region, selneig)
            # region = tf.transpose(region, [2, 0, 1, 3])

            ### shape = [batch, rand_num, 9, 256]

            # region = tf.reduce_mean(region, axis=2)
            region = tf.reshape(region, [-1, self.nei_num, 1, 256])
            region = slim.conv2d(region, 256, [self.nei_num, 1], stride=1, padding='VALID', activation_fn=tf.nn.relu,
                              weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                              normalizer_fn=slim.batch_norm, normalizer_params=bn_params,
                              scope='conv0')
            ### shape = [batch, rand_num, 256]
            _det = slim.fully_connected(tf.reshape(region, [-1, 256]), 1, activation_fn=tf.nn.sigmoid, scope='detect')
            _det = tf.reshape(_det, (tf.shape(_embed)[0], self.rand_num))


            return _det, randsel

    def detector_test(self, _embed, is_training=False):
        conv_init_params = tf.truncated_normal_initializer(stddev=0.02)
        fully_init_params = tf.random_normal_initializer(stddev=0.02)
        bias_init_params = tf.constant_initializer(0.0)
        bn_init_params = {'beta': tf.constant_initializer(0.),
                          'gamma': tf.random_normal_initializer(1., 0.02)}
        bn_params = {'is_training': is_training, 'decay': 0.9, 'epsilon': 1e-5,
                     'param_initializers': bn_init_params, 'updates_collections': None}



        with tf.variable_scope("detector") as scope:
            scope.reuse_variables()
            region = tf.transpose(_embed, [1, 0, 2])
            region = tf.transpose(tf.gather(region, tf.reshape(self.neighbor, [-1])), [1, 0, 2])
            region = tf.reshape(region, [-1, self.reg_num, self.nei_num, 256])
            # region = tf.transpose(_embed, [2, 1, 0, 3])
            # region = tf.tile(region, [self.reg_num, 1, 1, 1])
            # region = tf.gather_nd(region, self.n_idx)
            # region = tf.transpose(region, [2, 0, 1, 3])
            # shape = [batch, rand_num, 9, 256]
            # region = tf.reduce_mean(region, axis=2)
            region = tf.reshape(region, [-1, self.nei_num, 1, 256])
            region = slim.conv2d(region, 256, [self.nei_num, 1], stride=1, padding='VALID', activation_fn=tf.nn.relu,
                              weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                              normalizer_fn=slim.batch_norm, normalizer_params=bn_params,
                              scope='conv0')
            # shape = [batch, rand_num, 256]
            _det = slim.fully_connected(tf.reshape(region, [-1, 256]), 1, activation_fn=tf.nn.sigmoid, scope='detect')
            _det = tf.reshape(_det, (tf.shape(_embed)[0], self.reg_num))


            return _det


    def train(self, config):

        cwd = os.getcwd()

        optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss, var_list=self.vars)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        # self.sum = tf.summary.merge([self.loss_sum])
        self.sum = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(config.checkpoint_dir, self.sess.graph)

        # counter = 1

        load_dir = config.checkpoint_dir # + "/PASCAL_VOC_2012"
        if self.load(load_dir, config=config):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        testimg_loc = cwd + '/coco/%d/test/img_pre' % (self.image_size)
        testimg = os.listdir(testimg_loc)
        testimg.sort()
        testlabel_loc = cwd + '/coco/%d/test/knn_img/%d' % (self.image_size, self.nei_num)
        testlabel = os.listdir(testlabel_loc)
        testlabel.sort()
        testgt_loc = cwd + '/coco/%d/test/gt_img' % (self.image_size)
        testgt = os.listdir(testgt_loc)
        testgt.sort()
        testgt = testgt[16:21]

        ds_testimg = np.copy(testimg[16:21])  # for display, no shuffle
        ds_testlabel = np.copy(testlabel[16:21])  # for display, no shuffle


        for epoch in range(config.epoch):
            start_epoch = time.time()
            dataset = 1#randint(0, 1)
            if dataset == 0:
                # rot = randint(-3, 3)
                # flip = randint(0, 1)
                # if flip == 0:
                #     name = 'train'
                # else:
                #     name = 'flip'
                # name = 'flip'
                rot=0
                name="train"

                with tf.device('/cpu:1'):
                    trainimg_loc = cwd + '/PASCAL_VOC_2012/%d/r%d/%s_pre' % (self.image_size, rot, name)
                    trainimg = os.listdir(trainimg_loc)
                    trainimg.sort()
                    trainlabel_loc = cwd + '/PASCAL_VOC_2012/%d/r%d/knn/%s/%d' % (self.image_size, rot, name,self.nei_num)
                    trainlabel = os.listdir(trainlabel_loc)
                    trainlabel.sort()
                    traingt_loc = cwd + '/PASCAL_VOC_2012/%d/r%d/gt_%s' % (self.image_size, rot, name)
                    traingt = os.listdir(traingt_loc)
                    traingt.sort()
                    traingt = traingt[:5]

                    ds_trainimg = np.copy(trainimg[:5])  # for display, no shuffle
                    ds_trainlabel = np.copy(trainlabel[:5])  # for display, no shuffle
            else:
                flip = 0#randint(0, 1)
                if flip == 0:
                    name = 'img'
                else:
                    name = 'flip'

                with tf.device('/cpu:1'):
                    trainimg_loc = cwd + '/coco/%d/train/%s_pre' % (self.image_size, name)
                    trainimg = os.listdir(trainimg_loc)
                    trainimg.sort()
                    trainlabel_loc = cwd + '/coco/%d/train/knn_%s/%d' % (
                    self.image_size, name, self.nei_num)
                    trainlabel = os.listdir(trainlabel_loc)
                    trainlabel.sort()
                    traingt_loc = cwd + '/coco/%d/train/gt_%s' % (self.image_size, name)
                    traingt = os.listdir(traingt_loc)
                    traingt.sort()
                    traingt = traingt[:5]

                    ds_trainimg = np.copy(trainimg[:5])  # for display, no shuffle
                    ds_trainlabel = np.copy(trainlabel[:5])  # for display, no shuffle


            np.random.seed(epoch)
            np.random.shuffle(trainimg)
            np.random.seed(epoch)
            np.random.shuffle(trainlabel)

            batch_idxs = len(trainimg) // config.batch_size

            with tf.device('/cpu:1'):
                for idx in range(0, batch_idxs):
                    batch_files = []
                    batch_gt = []
                    for b in range(config.batch_size):
                        batch_files.append(np.load(trainimg_loc + '/' + trainimg[idx * config.batch_size + b]))
                        batch_gt.append(np.load(trainlabel_loc + '/' + trainlabel[idx * config.batch_size + b]).astype(float))

                # Update network
                self.sess.run(optim, feed_dict={self.input: batch_files, self.gt: batch_gt,
                                                self.is_training: True})


            testsize = 64
            np.random.seed(epoch)
            np.random.shuffle(testimg)
            np.random.seed(epoch)
            np.random.shuffle(testlabel)
            batch_files = []
            batch_label = []
            test_batch_files = []
            test_batch_label = []
            with tf.device('/cpu:1'):
                for b in range(testsize):
                    batch_files.append(np.load(trainimg_loc + '/' + trainimg[b]))
                    batch_label.append(np.load(trainlabel_loc + '/' + trainlabel[b]).astype(float))
                    test_batch_files.append(np.load(testimg_loc + '/' + testimg[b]))
                    test_batch_label.append(np.load(testlabel_loc + '/' + testlabel[b]).astype(float))
            self.display(epoch, config.epoch, batch_files, batch_label, test_batch_files,
                         test_batch_label, 1, self.counter)
            self.counter = self.counter + 1
            epoch_time = time.time() - start_epoch
            print("Epoch run time : %.9f"%(epoch_time))




            if np.mod(epoch, 10) == 1:
                # Visualization
                train_piece = []
                train_piece_label = []
                train_piece_gt = []
                test_piece = []
                test_piece_label = []
                test_piece_gt = []
                for b in range(5):
                    train_piece.append(np.load(trainimg_loc + '/' + ds_trainimg[b]))
                    train_piece_label.append(np.load(trainlabel_loc + '/' + ds_trainlabel[b]).astype(float))
                    train_piece_gt.append(np.load(traingt_loc + '/' + traingt[b]).astype(float))
                    test_piece.append(np.load(testimg_loc + '/' + ds_testimg[b]))
                    test_piece_label.append(np.load(testlabel_loc + '/' + ds_testlabel[b]).astype(float))
                    test_piece_gt.append(np.load(testgt_loc + '/' + testgt[b]).astype(float))
                det_vi_train = self.sess.run(self.det_test,
                                             feed_dict={self.input: train_piece, self.is_training: False})
                save_train = config.checkpoint_dir + "/train"
                if not os.path.exists(save_train):
                    os.makedirs(save_train)
                vi_train = Visualize(train_piece_gt, train_piece_label, det_vi_train, self.neighbor, save_train,
                                     epoch)
                vi_train.plot()

                det_vi_test = self.sess.run(self.det_test,
                                            feed_dict={self.input: test_piece, self.is_training: False})
                save_test = config.checkpoint_dir + "/test"
                if not os.path.exists(save_test):
                    os.makedirs(save_test)
                vi_test = Visualize(test_piece_gt, test_piece_label, det_vi_test, self.neighbor, save_test, epoch)
                vi_test.plot()



                self.save(config.checkpoint_dir, epoch, config)





    def display(self, _epoch, total_epochs, _trainimg, _trainlabel, _testimg, _testlabel, _idx, tb_c):
        # Caluculate match accurate for training set & test set

        # Calculate loss for training set & test set

        train_feeds = {self.input: _trainimg, self.gt: _trainlabel, self.is_training: False}
        train_loss, train_val, train_gt, train_score, train_tbloss, train_tbaccr \
            = self.sess.run([self.loss_test, self.det_full_test, self.gt_full_test, self.accr_t, self.loss_sum_train, self.accr_train],feed_dict=train_feeds)
        test_feeds = {self.input: _testimg, self.gt: _testlabel, self.is_training: False}

        start_time = time.time()
        test_loss, test_val, test_gt, test_score, test_tbloss, test_tbaccr \
            = self.sess.run([self.loss_test, self.det_full_test, self.gt_full_test, self.accr_t, self.loss_sum_test, self.accr_test],
                                                     feed_dict=test_feeds)
        run_time = time.time() - start_time

        print("Epoch: [%04d/%04d, batch : %04d] train accuracy: %.9f, train loss: %.9f, test accuracy: %.9f, "
              "test loss: %.9f, test running time: %.9f"
              % (_epoch + 1, total_epochs, _idx, train_score, train_loss, test_score, test_loss, run_time))

        self.writer.add_summary(train_tbloss, self.sess.run(tb_c))
        self.writer.add_summary(train_tbaccr, self.sess.run(tb_c))
        self.writer.add_summary(test_tbloss, self.sess.run(tb_c))
        self.writer.add_summary(test_tbaccr, self.sess.run(tb_c))




    def save(self, checkpoint_dir, step, config):
        model_name = "comp_AE.model"
        model_dir = "%s" % (config.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir, config):
        print(" [*] Reading checkpoints...")

        model_dir = "%s" % (config.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False








