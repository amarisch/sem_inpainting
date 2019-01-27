# ---------------------------------------------------------
# TensorFlow Semantic Image Inpainting Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------
import os
import time
import numpy as np
import tensorflow as tf

from dataset import Dataset
from inpaint_model import ModelInpaint
import poissonblending as poisson
import utils as utils


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.dataset = Dataset(self.flags, self.flags.dataset)
        self.model = ModelInpaint(self.sess, self.flags)

        self.sess.run(tf.global_variables_initializer())

    def _make_folders(self):

        self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)
        self.classifier_dir = "../../googlenet_tf/examples/callbacks_10"

        self.test_out_dir = "{}/inpaint/{}".format(self.flags.dataset, self.flags.mask_type)
        if not os.path.isdir(self.test_out_dir):
            os.makedirs(self.test_out_dir)
        dirs = os.listdir(self.test_out_dir)
        runs_id = np.array([item.replace('run_','') for item  in dirs if item.find('run_') > -1]).astype(np.int)
        if len(runs_id) == 0:
            dir_runs = 'run_0'
        else:
            dir_runs = 'run_' + str(max(runs_id)+1);

        self.test_out_dir = os.path.join(self.test_out_dir,dir_runs)
        os.makedirs(self.test_out_dir)

        with open(os.path.join(self.test_out_dir,'info.txt'),'w') as file:
            file.write('lambda_cat: ' + str(self.flags.lambda_cat) + '\n')
            file.write('lambda_inp: ' + str(self.flags.lambda_inp) + '\n')
            file.write('lambda_sem: ' + str(self.flags.lambda_sem) + '\n')
            file.write('layer_sem: ' + str(self.flags.layer) + '\n')
        #self.train_writer = tf.summary.FileWriter("{}/inpaint/{}/is_blend_{}/{}/log".format(
        #    self.flags.dataset, self.flags.load_model, str(self.flags.is_blend), self.flags.mask_type),
        #    graph_def=self.sess.graph_def)

    def test(self):
        if self.load_model():
            print(' [*] Load SUCCESS!')
        else:
            print(' [!] Load Failed...')


        loss_summary_all = []
        for num_try in range(self.flags.num_try):
            loss_summary = []
            self.model.preprocess(id_mask = self.dataset.batch_number)  # initialize memory and masks for inpaint model

            imgs = self.dataset.val_next_batch(batch_size=self.flags.sample_batch)  # select next image in the validation data
            best_loss = np.ones(self.flags.sample_batch) * 1e10
            best_outs = np.zeros_like(imgs)

            start_time = time.time()  # measure inference time
            for iter_time in range(self.flags.iters):
                loss, img_outs, summary = self.model(imgs, iter_time)  # inference
                loss_summary.append(loss)

                # save best gen_results accroding to the total loss
                for iter_loss in range(self.flags.sample_batch):
                    if best_loss[iter_loss] > loss[5][iter_loss]:  # total loss
                        best_loss[iter_loss] = loss[5][iter_loss]
                        best_outs[iter_loss] = img_outs[iter_loss]

                self.model.print_info(loss, iter_time, num_try)  # pring loss information

                #if num_try == 0:  # save first try-information on the tensorboard only
                #    self.train_writer.add_summary(summary, iter_time)  # write to tensorboard
                #    self.train_writer.flush()

            loss_summary_all.append(np.asarray(loss_summary))

            blend_results = self.postprocess(imgs, best_outs, self.flags.is_blend)  # blending

            total_time = time.time() - start_time
            print('Total PT: {:.3f} sec.'.format(total_time))

            img_list = [(imgs + 1.) / 2., blend_results, (imgs + 1.) / 2.]
            #pickle.dump(img_list,open('self.test_out_dir/image_matrix_{}'.format(num_try),'wb'))
            #self.model.plots(img_list, self.test_out_dir, num_try)  # save all of the images
            self.model.small_plots(img_list, self.test_out_dir, num_try)

        loss_summary_all = np.stack(loss_summary_all)
        with open(os.path.join(self.test_out_dir,'loss_summary.npy'),'wb') as file:
            np.save(file, np.asarray(loss_summary_all))

    def postprocess(self, ori_imgs, gen_imgs, is_blend=True):
        outputs = np.zeros_like(ori_imgs)
        tar_imgs = np.asarray([utils.inverse_transform(img) for img in ori_imgs])  # from (-1, 1) to (0, 1)
        sour_imgs = np.asarray([utils.inverse_transform(img) for img in gen_imgs])  # from (-1, 1) to (0, 1)

        if is_blend is True:
            for idx in range(tar_imgs.shape[0]):
                outputs[idx] = np.clip(poisson.blend(tar_imgs[idx], sour_imgs[idx],
                                                     ((1. - self.model.masks[idx]) * 255.).astype(np.uint8)), 0, 1)
        else:
            outputs = np.multiply(tar_imgs, self.model.masks) + np.multiply(sour_imgs, 1. - self.model.masks)

        return outputs

    def load_model(self):
        print(' [*] Reading checkpoint...')

        self._make_folders()
        self.iter_time = 0

        ckpt_dcgan = tf.train.get_checkpoint_state(self.model_out_dir)
        ckpt_dcgan_name = tf.train.latest_checkpoint(self.model_out_dir)
        ckpt_dcgan_variables = [item[0] for item in tf.contrib.framework.list_variables(ckpt_dcgan_name)]

        # ckpt_classifier_name = tf.train.latest_checkpoint(self.classifier_dir)
        ckpt_classifier_name = os.path.join(self.classifier_dir,'inception-cifar-epoch-195')
        ckpt_classifier_variables = [item[0] for item in tf.contrib.framework.list_variables(ckpt_classifier_name)]

        self.saver_dcgan = tf.train.Saver(var_list=[v for v in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
                                     if v.name.strip(':0') in ckpt_dcgan_variables])
        self.saver_classifier = tf.train.Saver(var_list=[v for v in tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
                                     if v.name.strip(':0') in ckpt_classifier_variables])


        if ckpt_dcgan and ckpt_classifier_name:

            self.saver_dcgan.restore(self.sess, os.path.join(ckpt_dcgan_name))
            print('Reading DCGAN checkpoint completed.')
            self.saver_classifier.restore(self.sess, ckpt_classifier_name)
            print('Reading {} checkpoint completed.'.format(self.flags.classifier))

            meta_graph_path = ckpt_dcgan.model_checkpoint_path + '.meta'
            self.iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

            print('===========================')
            print('   iter_time: {}'.format(self.iter_time))
            print('===========================')
            return True
        else:
            return False
