import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import os


def lrelu(x, alpha=0.2, max_value=None):
    return tf.maximum(x, alpha*x)

class AE(object):
    def __init__(self, sess, image_size=21, image_dim=300,
                        output_size=21, output_dim=1, batch_size=64):

        self.sess = sess
        self.image_size = image_size
        self.image_dim = image_dim
        self.output_size = output_size
        self.output_dim = output_dim
        # self.batch_size = batch_size
        self.build_model()


    def build_model(self):
        self.input = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.image_dim])
        self.gt = tf.placeholder(tf.float32, [None, self.output_size, self.output_size, self.output_dim])
        self.batch_size = tf.shape(self.input)[0]

        embed = self.encoder(self.input)
        gen = self.generator(embed)

        # L1 Loss
        self.loss = tf.reduce_mean(tf.abs(self.gt - gen))
        # self.loss = tf.nn.sigmoid_cross_entropy_with_logits(gen, self.gt)
        self.loss_sum = tf.summary.scalar("loss", self.loss)
        self.vars = tf.trainable_variables()
        self.saver = tf.train.Saver()


    def encoder(self, _im, is_training=True, reuse=False):
        conv_init_params = tf.truncated_normal_initializer(stddev=0.02)
        fully_init_params = tf.random_normal_initializer(stddev=0.02)
        bias_init_params = tf.constant_initializer(0.0)
        bn_init_params = {'beta': tf.constant_initializer(0.),
                          'gamma': tf.random_normal_initializer(1., 0.02)}
        bn_params = {'is_training': is_training, 'decay': 0.9, 'epsilon': 1e-5,
                     'param_initializers': bn_init_params, 'updates_collections': None}
        with tf.variable_scope("encoder") as scope:
            if reuse:
                scope.reuse_variables()
            # 21x21x300 -> 11x11x512
            _e0 = slim.conv2d(_im, 300, [3, 3], stride=2, activation_fn=lrelu,
                                  weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                  normalizer_fn=slim.batch_norm, normalizer_params=bn_params,
                                  scope='conv0')
            # 11x11x512 -> 6x6x512
            _e1 = slim.conv2d(_e0, 300, [3, 3], stride=2, activation_fn=lrelu,
                                  weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                  normalizer_fn=slim.batch_norm, normalizer_params=bn_params,
                                  scope='conv1')
            # 6x6x512 -> 3x3x512
            _e2 = slim.conv2d(_e1, 300, [3, 3], stride=2, activation_fn=lrelu,
                                  weights_initializer=conv_init_params, biases_initializer=bias_init_params,
                                  normalizer_fn=slim.batch_norm, normalizer_params=bn_params,
                                  scope='conv2')
            # 3x3x512 -> 1x1x1024
            _embed = slim.fully_connected(tf.reshape(_e2, [-1, 3*3*300]), 512, activation_fn=lrelu,
                                  weights_initializer=fully_init_params, biases_initializer=bias_init_params,
                                  normalizer_fn=slim.batch_norm, normalizer_params=bn_params, scope='fconv')

        return _embed


    def deconv(self, _input, _output_shape, _weight_shape, _bias_shape, _bn_params, _last=False):
        _weights = tf.get_variable("weights", _weight_shape, initializer=tf.random_normal_initializer())
        _biases = tf.get_variable("biases", _bias_shape, initializer=tf.constant_initializer(0.0))
        _deconv = tf.nn.conv2d_transpose(_input, _weights, _output_shape,
                                      strides=[1, 2, 2, 1], padding='SAME')
        _deconv = tf.nn.bias_add(_deconv, _biases)
        _bn = slim.batch_norm(_deconv, decay=_bn_params['decay'], epsilon=_bn_params['epsilon'],
                              param_initializers=_bn_params['param_initializers'],
                              updates_collections=_bn_params['updates_collections'])
        if ~(_last):
            return tf.nn.relu(_bn)
        else :
            return tf.nn.sigmoid(_bn)


    def generator(self, _embed, is_training=True, reuse=False):
        fully_init_params = tf.random_normal_initializer(stddev=0.02)
        bias_init_params = tf.constant_initializer(0.0)
        bn_init_params = {'beta': tf.constant_initializer(0.),
                          'gamma': tf.random_normal_initializer(1., 0.02)}
        bn_params = {'is_training': is_training, 'decay': 0.9, 'epsilon': 1e-5,
                     'param_initializers': bn_init_params, 'updates_collections': None}
        with tf.variable_scope("generater") as scope:
            if reuse:
                scope.reuse_variables()
            fc_d = 512
            _embed = slim.fully_connected(_embed, 3 * 3 * fc_d/2, activation_fn=None,
                                         weights_initializer=fully_init_params, biases_initializer=bias_init_params,
                                         scope='deconv0')
            _d0 = tf.reshape(_embed, [self.batch_size, 3, 3, fc_d/2])
            _bn_d0 = slim.batch_norm(_d0, decay=bn_params['decay'], epsilon=bn_params['epsilon'],
                                  param_initializers=bn_params['param_initializers'],
                                  updates_collections=bn_params['updates_collections'])
            with tf.variable_scope("deconv1"):
                _d1 = self.deconv(_bn_d0, [self.batch_size, 6, 6, fc_d/4], [3, 3, fc_d/4, fc_d/2], [fc_d/4], bn_params)
            with tf.variable_scope("deconv2"):
                _d2 = self.deconv(_d1, [self.batch_size, 11, 11, fc_d/8], [3, 3, fc_d/8, fc_d/4], [fc_d/8], bn_params)
            with tf.variable_scope("deconv3"):
                _out = self.deconv(_d2, [self.batch_size, 21, 21, 1], [3, 3, 1, fc_d/8], [1], bn_params, True)

        return _out


    def train(self, config):
        # config contains [learning_rate, batch_size, epoch, checkpoint_dir, beta1, dataset_name]

        # np.random.seed(220)

        cwd = os.getcwd()
        trainimg = np.load(cwd+'/PASCAL_VOC_2012/0.2/compoundData_train_aug.npy')
        trainlabel = np.load(cwd + '/PASCAL_VOC_2012/0.2/compoundData_seg_train_aug.npy').astype(float)
        ds_trainimg = np.copy(trainimg)         # for display, no shuffle
        ds_trainlabel = np.copy(trainlabel)     # for display, no shuffle
        testimg = np.load(cwd+'/PASCAL_VOC_2012/0.2/compoundData_test.npy')
        testlabel = np.load(cwd + '/PASCAL_VOC_2012/0.2/compoundData_seg_test.npy').astype(float)
        didx = np.random.randint(64, size=5)

        optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.loss, var_list=self.vars)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.sum = tf.summary.merge([self.loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1


        if self.load(config.checkpoint_dir, config=config):
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
                random_flip = np.random.random_sample()
                if random_flip > 0.5:
                    batch_files = np.fliplr(batch_files)
                    batch_gt = np.fliplr(batch_gt)

                # Update network
                _, summary_str = self.sess.run([optim, self.sum], feed_dict={self.input: batch_files, self.gt: batch_gt})
                self.writer.add_summary(summary_str, counter)

            if np.mod(epoch, 200) == 1:
                self.save(config.checkpoint_dir, epoch, config)

            # if np.mod(epoch, 50) == 0:
            self.display(epoch, config.epoch, ds_trainimg, ds_trainlabel, testimg, testlabel, didx, config.checkpoint_dir)



    def display(self, _epoch, total_epochs, _trainimg, _trainlabel, _testimg, _testlabel, _didx, save_dir):

        # Caluculate match accurate for training set & test set
        _train_xs = np.reshape(_trainimg, (-1, 21, 21, 300))
        _train_ys = np.reshape(_trainlabel, (-1, 21, 21, 1))
        _train_total = np.shape(_train_xs)[0]
        train_acc_embed = self.encoder(self.input, is_training=False, reuse=True)
        train_acc_gen = self.generator(train_acc_embed, is_training=False, reuse=True)
        img_gen = self.sess.run(train_acc_gen,
                                feed_dict={self.input: _train_xs})
        train_score = 0
        for idx_acc_train in xrange(_train_total):
            img_score = 0
            for r1 in xrange(21):
                for c1 in xrange(21):
                    if np.abs(img_gen[idx_acc_train, r1, c1, 0]-_train_ys[idx_acc_train, r1, c1, 0]) < 0.5:
                        img_score += 1
            img_score = img_score/(21.*21.)
            train_score = train_score + img_score
        train_score = train_score/_train_total

        _test_xs = np.reshape(_testimg, (-1, 21, 21, 300))
        _test_ys = np.reshape(_testlabel, (-1, 21, 21, 1))
        _test_total = np.shape(_test_xs)[0]
        test_acc_embed = self.encoder(self.input, is_training=False, reuse=True)
        test_acc_gen = self.generator(test_acc_embed, is_training=False, reuse=True)
        img_gen = self.sess.run(test_acc_gen,
                                feed_dict={self.input: _test_xs})
        test_score = 0
        for idx_acc_test in xrange(_test_total):
            test_img_score = 0
            for r2 in xrange(21):
                for c2 in xrange(21):
                    if np.abs(img_gen[idx_acc_test, r2, c2, 0] - _train_ys[idx_acc_test, r2, c2, 0]) < 0.5:
                        test_img_score += 1
            test_img_score = test_img_score / (21. * 21.)
            test_score = test_score + test_img_score
        test_score = test_score / _test_total

        # Calculate loss for training set & test set

        train_feeds = {self.input: _train_xs, self.gt: _train_ys}
        train_loss = self.sess.run(self.loss, feed_dict=train_feeds)
        test_feeds = {self.input: _test_xs, self.gt: _test_ys}
        test_loss = self.sess.run(self.loss, feed_dict=test_feeds)
        print("Epoch: [%04d/%04d] train accuracy: %.9f, train loss: %.9f, train accuracy: %.9f, test loss: %.9f" % (_epoch + 1, total_epochs, train_score, train_loss, test_score, test_loss))


        # Test image, Training image plotting
        if np.mod(_epoch,50) == 0:
            plt.figure(1)

            for dtest in range(5):
                d_embed = self.encoder(self.input, is_training=False, reuse=True)
                d_gen = self.generator(d_embed, is_training=False, reuse=True)
                c_out = self.sess.run(d_gen,
                                 feed_dict={self.input: np.reshape(_testimg[_didx[dtest], :, :, :], (-1, 21, 21, 300))})
                a = plt.subplot(2, 5, dtest + 1)
                a.matshow(np.reshape(_testlabel[_didx[dtest], :, :, :], (21, 21)), cmap='gray')
                b = plt.subplot(2, 5, dtest + 5 + 1)
                b.matshow(np.reshape(c_out, (21, 21)), cmap='gray')

            plt.title("TEST")
            test_dir = os.path.join(save_dir, "output_figure_recent/test")
            test_fig = "test_%04d" %(_epoch+1)
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            plt.figure(1).savefig(os.path.join(test_dir, test_fig))
            # plt.figure(1).savefig(os.getcwd() + '/data/output_figure/test/test_%04d' % (_epoch + 1))

            plt.figure(2)
            for dtrain in range(5):
                d_embed = self.encoder(self.input, is_training=False, reuse=True)
                d_gen = self.generator(d_embed, is_training=False, reuse=True)
                c_out = self.sess.run(d_gen,
                               feed_dict={self.input: np.reshape(_trainimg[_didx[dtrain], :, :, :], (-1, 21, 21, 300))})
                a = plt.subplot(2, 5, dtrain + 1)
                a.matshow(np.reshape(_trainlabel[_didx[dtrain], :, :, :], (21, 21)), cmap='gray')
                b = plt.subplot(2, 5, dtrain + 5 + 1)
                b.matshow(np.reshape(c_out, (21, 21)), cmap='gray')

            plt.title("TRAIN")
            train_dir = os.path.join(save_dir, "output_figure_recent/train")
            train_fig = "train_%04d" % (_epoch + 1)
            if not os.path.exists(train_dir):
                os.makedirs(train_dir)
            plt.figure(2).savefig(os.path.join(train_dir, train_fig))
            # plt.figure(2).savefig(os.getcwd() + '/data/output_figure/train/train_%04d' % (_epoch + 1))

    def save(self, checkpoint_dir, step, config):
        model_name = "comp_AE.model"
        model_dir = "%s_recent" % (config.dataset_name)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir, config):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (config.dataset_name, config.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False








