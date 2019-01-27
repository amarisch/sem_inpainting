import os
import sys
import platform
import argparse
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image

sys.path.append('../')
import loader as loader
from src.nets.googlenet import GoogLeNet_cifar
from src.helper.trainer import Trainer
from src.helper.evaluator import Evaluator
from src.dataflow.cifar import unpickle

parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=str, default='conv2_3x3',
                    help='name_of_layer')
parser.add_argument('--iterations', type=int, default=1000,
                    help='num_of_iterations')
parser.add_argument('--lr', type=float, default=0.05,
                    help='learning_rate')
FLAGS = parser.parse_args()

n_class = 196
IM_PATH = '../data/cars_predict/'
SAVE_PATH = 'best_callbacks/'
OUT_PATH = 'images_optimized/'
epoch_load = 196

data = unpickle('../data/cars/test_batch')

with open('layer_activation','rb') as file:
    layer_activation = pickle.load(file)

model_fixed = GoogLeNet_cifar(
    n_channel=3, n_class=n_class, bn=True, sub_imagenet_mean=False)
model_fixed.create_test_model()

model_variable = GoogLeNet_cifar(
    n_channel=3, n_class=n_class, bn=True, sub_imagenet_mean=False)
model_variable.create_variable_model()

loss_tensor = tf.reduce_sum(tf.square(model_fixed.layers[FLAGS.layer] - model_variable.layers[FLAGS.layer]))/np.mean(layer_activation[FLAGS.layer][:,0])


lr = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.AdamOptimizer(lr).minimize(loss_tensor, var_list=[model_variable.z])

path_ckpt = os.path.join('{}inception-cifar-epoch-{}'.format(SAVE_PATH, epoch_load))
ckpt_variables = [item[0] for item in tf.contrib.framework.list_variables(path_ckpt)]


with tf.Session() as sess:
    #restore variables saved in the checkpoint
    saver = tf.train.Saver(var_list=[v for v in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
                                     if v.name.strip(':0') in ckpt_variables])
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, path_ckpt)
    sess.run(model_variable.z.initializer)

    best_loss = np.inf

    save_folder = os.path.join(OUT_PATH,FLAGS.layer)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    file = open(os.path.join(save_folder,'losses.txt'),'w')

    for i in range(FLAGS.iterations):

        _, loss, image = sess.run([optimizer, loss_tensor, model_variable.image],
                    feed_dict={model_fixed.image: data['image'][0].reshape(1,64,64,3),
                               lr: FLAGS.lr*np.power(0.9977,i)})

        if loss < best_loss:
            im = Image.fromarray(image.reshape([64,64,3]).astype(np.uint8))
            im.save(os.path.join(save_folder,'image_optimized.jpg'),'jpeg')
            with open(os.path.join(save_folder,'losses.txt'),'a') as file:
                file.write('Iteration: {}, loss = {}\n'.format(i+1,loss))
