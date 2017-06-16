import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from visualize import Visualize
import os


def lrelu(x, alpha=0.2, max_value=None):
    return tf.maximum(x, alpha*x)

class DET(object):
    def __init__(self, sess, image_size=10, image_dim=3, eye_num= 441):

        self.sess = sess
        self.image_size = image_size
        self.image_dim = image_dim
        self.eye_num = eye_num
        self.neighbor = np.load(os.getcwd()+"/PASCAL_VOC_2012/NN_441_manual.npy")
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
        with tf.name_scope("loss") as scope:
            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt_full, logits=det_full))
            self.loss = tf.divide(tf.nn.l2_loss(self.gt_full - self.det_full), tf.to_float(tf.shape(self.det_full)[0]))
            # self.loss = tf.reduce_mean(tf.abs(self.gt_full - self.det_full))
            self.loss_sum_train = tf.summary.scalar("loss_train", self.loss)
            self.loss_sum_test = tf.summary.scalar("loss_test", self.loss)
        # self.corr = tf.equal(tf.arg_max(det_full,1), tf.arg_max(gt_full,1))
        self.corr = tf.equal(self.gt_full>0.5, self.det_full>0.5)
        # with tf.name_scope("accr") assc
        with tf.name_scope("accr") as scope:
            self.accr = tf.reduce_mean(tf.cast(self.corr, "float"))
            self.accr_train = tf.summary.scalar("accr_train", self.accr)
            self.accr_test = tf.summary.scalar("accr_test", self.accr)
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

        _embed_comp = tf.reshape(_embed, (-1, self.eye_num, 1, 256))

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
        randsel = np.random.choice(19*19, self.rand_num, replace=False)
        np.sort(randsel)
        selneig = np.take(self.n_idx,randsel, axis=0)
        for o in range(self.rand_num):
            for n in range(self.nei_num):
                selneig[o,n,0] = o

        with tf.variable_scope("detector") :
            region = tf.transpose(_embed, [2, 1, 0, 3])
            region = tf.tile(region, [self.rand_num, 1, 1, 1])
            region = tf.gather_nd(region, selneig)
            region = tf.transpose(region, [2, 0, 1, 3])
            # shape = [batch, rand_num, 9, 256]
            region = tf.reduce_mean(region, axis=2)
            # region = tf.reshape(region, [-1, 3, 3, 256])
            # region = slim.conv2d(region, 256, [3, 3], stride=1, padding='VALID', activation_fn=tf.nn.relu,
            #                   weights_initializer=conv_init_params, biases_initializer=bias_init_params,
            #                   normalizer_fn=slim.batch_norm, normalizer_params=bn_params,
            #                   scope='conv0')
            # shape = [batch, rand_num, 256]
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
            region = tf.transpose(_embed, [2, 1, 0, 3])
            region = tf.tile(region, [self.reg_num, 1, 1, 1])
            region = tf.gather_nd(region, self.n_idx)
            region = tf.transpose(region, [2, 0, 1, 3])
            # shape = [batch, rand_num, 9, 256]
            region = tf.reduce_mean(region, axis=2)
            # region = tf.reshape(region, [-1, 3, 3, 256])
            # region = slim.conv2d(region, 256, [3, 3], stride=1, padding='VALID', activation_fn=tf.nn.relu,
            #                   weights_initializer=conv_init_params, biases_initializer=bias_init_params,
            #                   normalizer_fn=slim.batch_norm, normalizer_params=bn_params,
            #                   scope='conv0')
            # shape = [batch, rand_num, 256]
            _det = slim.fully_connected(tf.reshape(region, [-1, 256]), 1, activation_fn=tf.nn.sigmoid, scope='detect')
            _det = tf.reshape(_det, (tf.shape(_embed)[0], self.reg_num))


            return _det


    def train(self, config):

        cwd = os.getcwd()
        with tf.device('/cpu:1'):
            trainimg = np.load(cwd+'/PASCAL_VOC_2012/10/compoundData_train.npy')
            trainlabel = np.load(cwd + '/PASCAL_VOC_2012/10/compoundData_seg_train_manual.npy').astype(float)
            ds_trainimg = np.copy(trainimg)       # for display, no shuffle
            ds_trainlabel = np.load(cwd + '/PASCAL_VOC_2012/10/compoundData_gt_train.npy').astype(float)     # for display, no shuffle
            testimg = np.load(cwd+'/PASCAL_VOC_2012/10/compoundData_test.npy')
            testlabel = np.load(cwd + '/PASCAL_VOC_2012/10/compoundData_seg_test_manual.npy').astype(float)
            ds_testlabel = np.load(cwd + '/PASCAL_VOC_2012/10/compoundData_gt_test.npy').astype(float)
        # didx = np.random.randint(64, size=5)

        optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss, var_list=self.vars)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        # self.sum = tf.summary.merge([self.loss_sum])
        self.sum = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(config.checkpoint_dir, self.sess.graph)

        counter = 1

        load_dir = config.checkpoint_dir # + "/PASCAL_VOC_2012"
        if self.load(load_dir, config=config):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")


        for epoch in range(config.epoch):
            np.random.seed(epoch)
            np.random.shuffle(trainimg)
            np.random.seed(epoch)
            np.random.shuffle(trainlabel)

            batch_idxs = len(trainimg) // config.batch_size


            for idx in range(0, batch_idxs):
                batch_files = trainimg[range(idx * config.batch_size, (idx + 1) * config.batch_size)]
                batch_gt = trainlabel[range(idx * config.batch_size, (idx + 1) * config.batch_size)]
                didx = np.random.randint(len(testimg), size=config.batch_size)
                test_batch_files = testimg[didx]
                test_batch_gt = testlabel[didx]
                random_flip = np.random.random_sample()
                if random_flip > 0.5:
                    batch_files = np.flip(batch_files, 1)
                    batch_gt = np.flip(batch_gt, 1)
                # Update network
                self.sess.run(optim, feed_dict={self.input: batch_files, self.gt: batch_gt,
                                                self.is_training: True})
                if np.mod(idx, 10) ==1:
                    self.display(epoch, config.epoch, batch_files, batch_gt, test_batch_files, test_batch_gt, idx, config.checkpoint_dir)
                    loss_train, accr_train = self.sess.run([self.loss_sum_train, self.accr_train], feed_dict={self.input: batch_files, self.gt: batch_gt,
                                                            self.is_training: False})
                    self.writer.add_summary(loss_train, counter)
                    self.writer.add_summary(accr_train, counter)

                    loss_test, accr_test = self.sess.run([self.loss_sum_test, self.accr_test],
                                                           feed_dict={self.input: test_batch_files, self.gt: test_batch_gt,
                                                                      self.is_training: False})
                    self.writer.add_summary(loss_test, counter)
                    self.writer.add_summary(accr_test, counter)
                    counter =counter +1


            # Visualization
            det_vi_train = self.sess.run(self.det_test, feed_dict={self.input:ds_trainimg[0:5], self.is_training: False})
            save_train = config.checkpoint_dir+"/train"
            if not os.path.exists(save_train):
                os.makedirs(save_train)
            vi_train = Visualize(ds_trainlabel[:5], det_vi_train, self.neighbor, save_train, epoch)
            vi_train.plot()


            det_vi_test = self.sess.run(self.det_test,feed_dict={self.input: testimg[0:5], self.is_training: False})
            save_test = config.checkpoint_dir + "/test"
            if not os.path.exists(save_test):
                os.makedirs(save_test)
            vi_test = Visualize(ds_testlabel[:5], det_vi_test, self.neighbor, save_test, epoch)
            vi_test.plot()


            if np.mod(epoch, 50) == 1:
                self.save(config.checkpoint_dir, epoch, config)




    def display(self, _epoch, total_epochs, _trainimg, _trainlabel, _testimg, _testlabel, _idx, save_dir):
        # Caluculate match accurate for training set & test set

        # Calculate loss for training set & test set

        train_feeds = {self.input: _trainimg, self.gt: _trainlabel, self.is_training: False}
        train_loss, train_val, train_gt, train_score = self.sess.run([self.loss, self.det_full, self.gt_full, self.accr],
                                                                     feed_dict=train_feeds)
        test_feeds = {self.input: _testimg, self.gt: _testlabel, self.is_training: False}
        test_loss, test_val, test_gt, test_score = self.sess.run([self.loss, self.det_full, self.gt_full, self.accr],
                                                     feed_dict=test_feeds)
        print("Epoch: [%04d/%04d, batch : %04d] train accuracy: %.9f, train loss: %.9f, test accuracy: %.9f, test loss: %.9f"
              % (_epoch + 1, total_epochs, _idx, train_score, train_loss, test_score, test_loss))




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








