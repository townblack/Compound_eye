import os
import scipy.misc
import numpy as np

from comp_AE_model import AE
# from utils import pp

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 5001, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_string("dataset_name", "PASCAL_VOC_2012", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the samples [samples]")
flags.DEFINE_string("checkpoint_dir", "output_enc/recent", "Directory name to save the samples [checkpoint]")
flags.DEFINE_string("test_dir", "test", "Directory name to save the samples [checkpoint]")
FLAGS = flags.FLAGS


with tf.device('/gpu:2'):
    def main(_):
        # pp.pprint(flags.FLAGS.__flags)

        # if not os.path.exists(FLAGS.sample_dir):
        #     os.makedirs(FLAGS.sample_dir)
        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)
        # if not os.path.exists(FLAGS.test_dir):
        #     os.makedirs(FLAGS.test_dir)

        aconfig = tf.ConfigProto()
        aconfig.gpu_options.allow_growth = True
        aconfig.allow_soft_placement = True
        # aconfig.gpu_options.per_process_gpu_memory_fraction = 0.5

        with tf.Session(config=aconfig) as sess:
            ae = AE(sess)
            ae.train(FLAGS)

    if __name__ == '__main__':
        tf.app.run()