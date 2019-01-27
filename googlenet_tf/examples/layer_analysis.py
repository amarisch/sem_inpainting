import os
import sys
import platform
import argparse
import numpy as np
import tensorflow as tf
import pickle

sys.path.append('../')
import loader as loader
from src.nets.googlenet import GoogLeNet
from src.dataflow.cifar import CIFAR
from src.dataflow.cifar import unpickle

n_class = 196
IM_PATH = '../data/cars_predict/'
SAVE_PATH = 'callbacks_10/'
epoch_load = 195

DATA_PATH = '../data/cars/'
image_set_size = 1629

model = GoogLeNet(
    n_channel=3, n_class=n_class, bn=True, sub_imagenet_mean=False)
model.create_test_model()

# valid_data = CIFAR(
#     data_dir=DATA_PATH,
#     shuffle=False,
#     batch_dict_name=['image', 'label'],
#     data_type='valid',
#     channel_mean=None,
#     subtract_mean=True,
#     augment=False,
#     # pf=pf_test,
#     )
# valid_data.setup(epoch_val=0, batch_size=image_set_size)
#
# data = valid_data.next_batch_dict()

def subtract_channel_mean(im_list):
    """
    Args:
        im_list: [batch, h, w, c]
    """
    mean = [107.96115631234653, 102.3883406590508, 103.1299406628875]
    for c_id in range(0, im_list.shape[-1]):
        im_list[:, :, :, c_id] = im_list[:, :, :, c_id] - mean[c_id]
    return im_list



data = unpickle('../data/cars/test_batch')

#data['image'] = subtract_channel_mean(data['image'])

#N = len(data['label'])
N = 100
correct_preds = 0
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, '{}inception-cifar-epoch-{}'.format(SAVE_PATH, epoch_load))

    for id in range(N):
        logits = sess.run(model.layers['logits'],
                        feed_dict={model.image: data['image'][id].reshape(1,64,64,3)})
        print('id = {}, predicted label: {}, true label: {}'.format(id, np.argmax(logits),data['label'][id]))
        if np.argmax(logits) == data['label'][id]:
            correct_preds += 1

print('Accuracy on the validation set: {}'.format(correct_preds/N))
