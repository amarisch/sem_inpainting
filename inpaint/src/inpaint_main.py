# ---------------------------------------------------------
# TensorFlow Semantic Image Inpainting Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import tensorflow as tf

from inpaint_solver import Solver

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_string('dataset', 'cars', 'dataset name for choice [celebA|svhn], default: celebA')
tf.flags.DEFINE_string('classifier', 'googlenet', 'name of the clssifier used for semantic loss')
tf.flags.DEFINE_bool('training','false','no training dataset needed for inpainting')
tf.flags.DEFINE_float('learning_rate', 0.01, 'learning rate to update latent vector z, default: 0.01') # 0.01 probably used by moodoki
tf.flags.DEFINE_float('momentum', 0.9, 'momentum term of the NAG optimizer for latent vector, default: 0.9') # 0.9 used by moodoki
tf.flags.DEFINE_integer('z_dim', 100, 'dimension of z vector, default: 100') # 100 used by Yeh
tf.flags.DEFINE_float('lamb', 0.003, 'hyper-parameter for prior loss, default: 0.1')  # 0.003 used by Yeh, 0.1 used by moodoki
tf.flags.DEFINE_bool('is_blend', True, 'blend predicted image to original image, default: true')
tf.flags.DEFINE_float('lambda_sem', 1e5, 'weight of semantic loss')
tf.flags.DEFINE_string('layer', 'inception_4b', 'layer chosen for semantic loss')
tf.flags.DEFINE_string('layer_2', 'inception_3b', '2nd layer chosen for semantic loss')
tf.flags.DEFINE_float('lambda_cat', 1e6, 'weight of categorical loss')
tf.flags.DEFINE_float('lambda_inp', 3, 'weight of inpainting loss')
tf.flags.DEFINE_string('mask_type', 'center', 'mask type choice in [center|left|random], default: center')
tf.flags.DEFINE_integer('img_size', 64, 'image height or width, default: 64')

tf.flags.DEFINE_integer('iters', 1500, 'number of iterations to optimize latent vector, default: 1500') # 1'500 used by Yeh
tf.flags.DEFINE_integer('num_try', 50, 'number of randome samples, default: 20')
tf.flags.DEFINE_integer('print_freq', 100, 'print frequency for loss, default: 100')
tf.flags.DEFINE_integer('sample_batch', 1, 'number of sampling images, default: 2')
tf.flags.DEFINE_string('load_model', '20190111-0942',
                       'saved DCGAN model that you wish to test, (e.g. 20180704-1736), default: None')
tf.flags.DEFINE_integer('n_class', 196,
                       'number of GoogleNet classification classes, default: 196')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index\

    solver = Solver(FLAGS)
    solver.test()



if __name__ == '__main__':
    tf.app.run()
