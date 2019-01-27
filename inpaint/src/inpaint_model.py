import collections
import numpy as np
import tensorflow as tf
from scipy.signal import convolve2d
import matplotlib as mpl
mpl.use('Agg')  # or whatever other backend that you want to solve Segmentation fault (core dumped)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
from PIL import Image

import utils as utils
from dcgan import DCGAN
from mask_generator import load_mask, gen_mask

import sys
sys.path.append('../../googlenet_tf')
from src.nets.googlenet import GoogLeNet


class ModelInpaint(object):
    def __init__(self, sess, flags):
        self.sess = sess
        self.flags = flags
        print(self.flags.n_class)
        self.image_size = (flags.img_size, flags.img_size, 3)

        self.z_vectors, self.learning_rate, self.velocity = None, None, None
        self.masks, self.wmasks = None, None

        self.dcgan = DCGAN(sess, Flags(flags), self.image_size)

        self.classifier_gen = GoogLeNet(n_channel=3, n_class=flags.n_class, bn=True, sub_imagenet_mean=False)
        self.classifier_inp = GoogLeNet(n_channel=3, n_class=flags.n_class, bn=True, sub_imagenet_mean=False)


        self._build_net()
        self._tensorboard()




        print('Initialized Model Inpaint SUCCESS!')

    def _build_net(self):
        self.wmasks_ph = tf.placeholder(tf.float32, [None, *self.image_size], name='wmasks')
        self.images_ph = tf.placeholder(tf.float32, [None, *self.image_size], name='images')
        self.masks_ph = tf.placeholder(tf.float32, [None, *self.image_size], name='masks')

        self.masked_image = tf.multiply(self.masks_ph,self.images_ph)
        self.inp_image = tf.multiply(self.masks_ph,self.images_ph) + tf.multiply(1. - self.masks_ph, self.dcgan.g_samples)

        self.context_loss = tf.reduce_sum(tf.contrib.layers.flatten(
            tf.abs(tf.multiply(self.wmasks_ph, self.dcgan.g_samples) - tf.multiply(self.wmasks_ph, self.images_ph))), 1)
        self.prior_loss = tf.squeeze(self.flags.lamb * self.dcgan.g_loss_without_mean)  # from (2, 1) to (2,)

        lambda_inp = tf.constant(self.flags.lambda_inp,dtype = tf.float32)
        lambda_cat = tf.constant(self.flags.lambda_cat,dtype = tf.float32)
        lambda_sem = tf.constant(self.flags.lambda_sem,dtype = tf.float32)

        self.dcgan.build_inpainting_discriminator(self.inp_image)
        self.inp_loss = self.dcgan.d_loss_inpainted

        #self.classifier_gen.create_inpainting_model(self.sub_channel_mean(self.dcgan.g_samples))
        #self.classifier_inp.create_inpainting_model(self.sub_channel_mean(self.inp_image))

        self.classifier_gen.create_inpainting_model(self.dcgan.g_samples)
        self.classifier_inp.create_inpainting_model(self.inp_image)

        self.semantic_loss = tf.reduce_sum(tf.contrib.layers.flatten(tf.square(self.classifier_gen.layers[self.flags.layer] -
                            self.classifier_inp.layers[self.flags.layer])), 1)/tf.reduce_sum(tf.contrib.layers.flatten(tf.square(
                            self.classifier_inp.layers[self.flags.layer])), 1) + tf.reduce_sum(tf.contrib.layers.flatten(tf.square(self.classifier_gen.layers[self.flags.layer_2] -
                                                self.classifier_inp.layers[self.flags.layer_2])), 1)/tf.reduce_sum(tf.contrib.layers.flatten(tf.square(
                                                self.classifier_inp.layers[self.flags.layer_2])), 1)

        self.categorical_loss = self.Jensen_div(tf.nn.softmax(self.classifier_gen.layers['logits']),tf.nn.softmax(self.classifier_inp.layers['logits']))

        self.total_loss = self.context_loss + self.prior_loss + lambda_inp*self.inp_loss + lambda_sem*self.semantic_loss + lambda_cat*self.categorical_loss
        #self.total_loss = self.context_loss + self.prior_loss

        self.grad = tf.gradients(self.total_loss, self.dcgan.z)

    def Jensen_div(self, a, b):
        return tf.reduce_sum(tf.multiply(a, tf.log(a/b)) + tf.multiply(b, tf.log(b/a)), 1)

    def sub_channel_mean(self, inputs):
        with tf.name_scope('sub_mean'):
            red, green, blue = tf.split(axis=3,
                                        num_or_size_splits=3,
                                        value=inputs)

            mean = [107.96115631234653, 102.3883406590508, 103.1299406628875]

            input_m = tf.concat(axis=3, values=[
                red - mean[0],
                green - mean[1],
                blue - mean[2],
            ])
            return input_m

    def preprocess(self, id_mask = 0, use_weighted_mask=True, nsize=7):
        self.z_vectors = np.random.randn(self.flags.sample_batch, self.flags.z_dim)

        self.masks = load_mask(self.flags, id = id_mask)
        self.learning_rate = self.flags.learning_rate
        self.velocity = 0.  # for latent vector optimization

        if use_weighted_mask is True:
            wmasks = self.create_weighted_mask(self.masks, nsize)
        else:
            wmasks = self.masks

        self.wmasks = self.create3_channel_masks(wmasks)
        self.masks = self.create3_channel_masks(self.masks)

    def subtract_channel_mean(self, im_list):
        """
        Args:
            im_list: [batch, h, w, c]
        """
        mean = [107.96115631234653, 102.3883406590508, 103.1299406628875]
        for c_id in range(0, im_list.shape[-1]):
            im_list[:, :, :, c_id] = im_list[:, :, :, c_id] - mean[c_id]
        return im_list

    def _tensorboard(self):
        tf.summary.scalar('loss/context_loss', tf.reduce_mean(self.context_loss))
        tf.summary.scalar('loss/prior_loss', tf.reduce_mean(self.prior_loss))
        tf.summary.scalar('loss/total_loss', tf.reduce_mean(self.total_loss))

        self.summary_op = tf.summary.merge_all()

    def __call__(self, imgs, iter_time):
        feed_dict = {self.dcgan.z: self.z_vectors,
                     self.wmasks_ph: self.wmasks,
                     self.images_ph: imgs,
                     self.masks_ph: self.masks}

        out_vars = [self.context_loss, self.prior_loss, self.categorical_loss, self.inp_loss, self.semantic_loss,
            self.total_loss, self.grad, self.dcgan.g_samples, self.summary_op]


        #out_vars = [self.context_loss, self.prior_loss, tf.constant(0,dtype=tf.float32), tf.constant(0,dtype=tf.float32), tf.constant(0,dtype=tf.float32),
        #            self.total_loss, self.grad, self.dcgan.g_samples, self.summary_op]

        # if not flags.inp_loss == 0:
        #     out_vars[2] = self.inp_loss
        # if not flags.categorical_loss == 0:
        #     out_vars[2] = self.categorical_loss
        # if not flags.semantic_loss == 0:
        #     out_vars[3] = self.semantic_loss

        context_loss, prior_loss, categorical_loss, inp_loss, semantic_loss, total_loss, grad, img_out, summary = self.sess.run(out_vars, feed_dict=feed_dict)

        # learning rate control
        if np.mod(iter_time, 100) == 0:
            self.learning_rate *= 0.95

        # Nesterov Acceleratd Gradient (NAG)
        v_prev = np.copy(self.velocity)
        self.velocity = self.flags.momentum * self.velocity - self.learning_rate * grad[0]
        self.z_vectors += -self.flags.momentum * v_prev + (1 + self.flags.momentum) * self.velocity
        self.z_vectors = np.clip(self.z_vectors, -1., 1.)  # as paper mentioned

        return [context_loss, prior_loss, categorical_loss, inp_loss, semantic_loss, total_loss], img_out, summary

    def print_info(self, loss, iter_time, num_try):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([('num_try', num_try), ('tar_try', self.flags.num_try),
                                                  ('cur_iter', iter_time), ('tar_iters', self.flags.iters),
                                                  ('batch_size', self.flags.sample_batch),
                                                  ('context_loss', np.mean(loss[0])),
                                                  ('prior_loss', np.mean(loss[1])),
                                                  ('categorical_loss', self.flags.lambda_cat*np.mean(loss[2])),
                                                  ('inpainting_loss', self.flags.lambda_inp*np.mean(loss[3])),
                                                  ('semantic_loss', self.flags.lambda_sem*np.mean(loss[4])),
                                                  ('total_loss', np.mean(loss[5])),
                                                  ('mask_type', self.flags.mask_type),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_time, ord_output)

    @staticmethod
    def create_weighted_mask(masks, nsize):
        wmasks = np.zeros_like(masks)
        ker = np.ones((nsize, nsize), dtype=np.float32)
        ker = ker / np.sum(ker)

        for idx in range(masks.shape[0]):
            mask = masks[idx]
            inv_mask = 1. - mask
            temp = mask * convolve2d(inv_mask, ker, mode='same', boundary='symm')
            wmasks[idx] = mask * temp

        return wmasks

    @staticmethod
    def create3_channel_masks(masks):
        masks_3c = np.zeros((*masks.shape, 3), dtype=np.float32)

        for idx in range(masks.shape[0]):
            mask = masks[idx]
            masks_3c[idx, :, :, :] = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        return masks_3c

    def plots(self, img_list, save_file, num_try):
        n_cols = len(img_list)
        n_rows = self.flags.sample_batch

        # parameters for plot size
        scale, margin = 0.04, 0.001
        cell_size_h, cell_size_w = img_list[0][0].shape[0] * scale, img_list[0][0].shape[1] * scale
        fig = plt.figure(figsize=(cell_size_w * n_cols, cell_size_h * n_rows))  # (column, row)
        gs = gridspec.GridSpec(n_rows, n_cols)  # (row, column)
        gs.update(wspace=margin, hspace=margin)

        # save more bigger image
        for col_index in range(n_cols):
            for row_index in range(n_rows):
                ax = plt.subplot(gs[row_index * n_cols + col_index])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')

                if col_index == 0:  # original input image
                    plt.imshow((img_list[col_index][row_index] * self.masks[row_index]).reshape(
                        self.image_size[0], self.image_size[1], self.image_size[2]), cmap='Greys_r')
                else:
                    plt.imshow((img_list[col_index][row_index]).reshape(
                        self.image_size[0], self.image_size[1], self.image_size[2]), cmap='Greys_r')

        plt.savefig(save_file + '/{}_{}.png'.format(self.flags.mask_type, num_try), bbox_inches='tight')
        plt.close(fig)

    def small_plots(self, img_list, save_file, num_try):
        img_save = Image.fromarray((np.concatenate([img_list[0].squeeze()*self.masks[0],
                img_list[1].squeeze(),img_list[2].squeeze()],axis = 1)*255).astype(np.uint8))
        img_save.save(save_file + '/{}_{}.png'.format(self.flags.mask_type, num_try),'png')


class Flags(object):
    def __init__(self, flags):
        self.z_dim = flags.z_dim
        self.learning_rate = flags.learning_rate
        self.beta1 = flags.momentum
        self.sample_batch = flags.sample_batch
