import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import os


def lrelu(x, alpha=0.2, max_value=None):
    return tf.maximum(x, alpha*x)

class DET(object):
    def __init__(self, sess, image_size=10, image_dim=3, eye_num= 441, det_stride = 5, batch_size=64):

        self.sess = sess
        self.image_size = image_size
        self.image_dim = image_dim
        self.eye_num = eye_num
        self.neighbor = np.load(os.getcwd()+"/PASCAL_VOC_2012/NN_441.npy")
        n_idx = []
        for s in range(eye_num):
            c_idx = []
            for n in range(9):
                c_idx.append([s, self.neighbor[s][n]])
            n_idx.append(c_idx)
        self.n_idx = n_idx
        # self.batch_size = batch_size
        self.build_model()


    def build_model(self):
        self.input = tf.placeholder(tf.float32, [None, self.eye_num, self.image_size, self.image_size, self.image_dim])
        self.gt = tf.placeholder(tf.float32, [None, self.eye_num])
        self.batch_size = tf.shape(self.input)[0]
        self.is_training = tf.placeholder(tf.bool)

        self.enc = self.encoder(self.input, self.is_training)
        self.det = self.detector(self.enc)


        # L1 Loss
        # self.loss = tf.reduce_mean(tf.abs(self.gt - self.det))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.gt, logits=self.det))
        self.loss_sum = tf.summary.scalar("loss", self.loss)
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
            _e0 = slim.conv2d(_cells, 32, [3, 3], stride=2, activation_fn=tf.nn.relu,
                                  weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                  normalizer_fn=slim.batch_norm, normalizer_params=bn_params,
                                  scope='conv0')
            _e1 = slim.conv2d(_e0, 64, [3, 3], stride=2, activation_fn=tf.nn.relu,
                                  weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                  normalizer_fn=slim.batch_norm, normalizer_params=bn_params,
                                  scope='conv1')
            _embed = slim.fully_connected(tf.reshape(_e1, [-1, 3*3*64]), 128, activation_fn=tf.nn.relu,
                                  weights_initializer=fully_init_params, biases_initializer=bias_init_params,
                                  normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='fconv')

        # full batch -> batch, 441

        _embed_comp = tf.reshape(_embed, (-1, self.eye_num, 1, 128))

        return _embed_comp

    def detector(self, _embed):
        with tf.variable_scope("detector") :
            region = tf.tile(_embed, [1, 1, self.eye_num, 1])
            region = tf.transpose(region, [2, 1, 0, 3])
            region = tf.gather_nd(region, self.n_idx)
            region = tf.transpose(region, [2, 0, 1, 3])
            # shape = [batch, 441, 9, 128]
            region = tf.reduce_mean(region, axis=2)

            _det = slim.fully_connected(tf.reshape(region, [-1, 128]), 1, activation_fn=None, scope='detect')
            _det = tf.reshape(_det, (tf.shape(_embed)[0], self.eye_num))
            return _det


    def train(self, config):

        cwd = os.getcwd()
        with tf.device('/cpu:1'):
            trainimg = np.load(cwd+'/PASCAL_VOC_2012/compoundData_train.npy')
            trainlabel = np.load(cwd + '/PASCAL_VOC_2012/compoundData_seg_train.npy').astype(float)
            # ds_trainimg = np.reshape(np.copy(trainimg), (-1, 21, 21,300))         # for display, no shuffle
            # ds_trainlabel = np.reshape(np.copy(trainlabel),(-1, 21, 21, 1))     # for display, no shuffle
            testimg = np.load(cwd+'/PASCAL_VOC_2012/compoundData_test.npy')
            testlabel = np.load(cwd + '/PASCAL_VOC_2012/compoundData_seg_test.npy').astype(float)
        # didx = np.random.randint(64, size=5)
        didx = [5,6,7,8,9]

        optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss, var_list=self.vars)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.sum = tf.summary.merge([self.loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1

        load_dir = config.checkpoint_dir + "/PASCAL_VOC_2012"
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
                didx = np.random.randint(len(testimg), size=30)
                test_batch_files = testimg[didx]
                test_batch_gt = testlabel[didx]
                # batch_gt = batch_gt[:,self.stride_component,:]
                # random_flip = np.random.random_sample()
                # if random_flip > 0.5:
                #     for batch in range(config.batch_size):
                #         batch_files[batch, :, :, :, :] = np.fliplr(batch_files[batch, :, :, :, :])
                #         batch_gt[batch, :, :, :, :] = np.fliplr(batch_gt[batch, :, :, :, :])

                    # batch_files = np.fliplr(batch_files)
                    # batch_gt = np.fliplr(batch_gt)

                # Update network
                _, summary_str = self.sess.run([optim, self.sum], feed_dict={self.input: batch_files, self.gt: batch_gt,
                                                                             self.is_training: True})
                self.writer.add_summary(summary_str, counter)
                # if np.mod(idx, 300) ==1:
                #     self.display(epoch, config.epoch, batch_files, batch_gt, test_batch_files, test_batch_gt, idx, config.checkpoint_dir)
            if np.mod(epoch, 200) == 1:
                self.save(config.checkpoint_dir, epoch, config)

            # if np.mod(epoch, 50) == 0:
            self.display(epoch, config.epoch, trainimg[:30], trainlabel[:30], testimg[:30], testlabel[:30], 0, config.checkpoint_dir)



    def display(self, _epoch, total_epochs, _trainimg, _trainlabel, _testimg, _testlabel, _idx, save_dir):
        with tf.device('/gpu:2'):
        # Caluculate match accurate for training set & test set
            _train_total = np.shape(_trainimg)[0]

            #
            # _trainimg = _trainimg[0:30]
            # _trainlabel = _trainlabel[0:30]
            train_det, train_loss = self.sess.run([self.det, self.loss],
                                                      feed_dict={self.input: _trainimg, self.gt: _trainlabel,
                                                                  self.is_training: False})


            train_score = 0
            for idx_acc_train in range(_train_total):
                img_score = 0
                for eye in range(self.eye_num):
                    if np.abs(train_det[idx_acc_train, eye]-_trainlabel[idx_acc_train, eye]) < 0.7:
                        img_score += 1.
                img_score = img_score/(np.float(self.eye_num))
                train_score = train_score + img_score
            train_score = train_score/(_train_total)


            _test_total = np.shape(_testimg)[0]
            # _testimg = _testimg[0:30]
            # _testlabel = _testlabel[0:30]
            # _testlabel = _testlabel[:, self.stride_component, :]
            test_det, test_loss = self.sess.run([self.det, self.loss],
                                                    feed_dict={self.input: _testimg, self.gt: _testlabel,
                                                                self.is_training: False})
            test_score = 0
            for idx_acc_test in range(_test_total):
                test_img_score = 0
                for eye in range(self.eye_num):
                    if np.abs(test_det[idx_acc_test, eye]-_testlabel[idx_acc_test, eye]) < 0.7:
                        test_img_score += 1.
                test_img_score = test_img_score/(np.float(self.eye_num))
                test_score = test_score + test_img_score
            test_score = test_score / _test_total

            # Calculate loss for training set & test set

            train_feeds = {self.input: _trainimg, self.gt: _trainlabel, self.is_training: False}
            train_loss = self.sess.run(self.loss, feed_dict=train_feeds)
            test_feeds = {self.input: _testimg, self.gt: _testlabel, self.is_training: False}
            test_loss = self.sess.run(self.loss, feed_dict=test_feeds)
            print("Epoch: [%04d/%04d, batch : %04d] train accuracy: %.9f, train loss: %.9f, test accuracy: %.9f, test loss: %.9f"
                  % (_epoch + 1, total_epochs, _idx, train_score, train_loss, test_score, test_loss))


        # Test image, Training image plotting
        # if np.mod(_epoch,20) == 0:
        #
        #     plt.figure(1)
        #     plt.title("TRAIN")
        #
        #     for dtrain in range(5):
        #         c_out = train_img_gen[_didx[dtrain]]
        #         a = plt.subplot(2, 5, dtrain + 1)
        #         a.matshow(np.reshape(_trainlabel[_didx[dtrain], :, :, :], (21, 21)), cmap='gray')
        #         b = plt.subplot(2, 5, dtrain + 5 + 1)
        #         b.matshow(np.reshape(c_out, (21, 21)), cmap='gray')
        #
        #     test_dir = os.path.join(save_dir, "output_figure/train")
        #     test_fig = "train%04d" % (_epoch + 1)
        #     if not os.path.exists(test_dir):
        #         os.makedirs(test_dir)
        #     plt.figure(1).savefig(os.path.join(test_dir, test_fig))
        #
        #
        #     plt.figure(2)
        #     plt.title("TEST")
        #
        #     for dtest in range(5):
        #         c_out = test_img_gen[_didx[dtest]]
        #         a = plt.subplot(2, 5, dtest + 1)
        #         a.matshow(np.reshape(_testlabel[_didx[dtest], :, :, :], (21, 21)), cmap='gray')
        #         b = plt.subplot(2, 5, dtest + 5 + 1)
        #         b.matshow(np.reshape(c_out, (21, 21)), cmap='gray')
        #
        #     test_dir = os.path.join(save_dir, "output_figure/test")
        #     test_fig = "test_test%04d" % (_epoch + 1)
        #     if not os.path.exists(test_dir):
        #         os.makedirs(test_dir)
        #     plt.figure(2).savefig(os.path.join(test_dir, test_fig))



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








