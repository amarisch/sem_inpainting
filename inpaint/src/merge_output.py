import os
import cv2
import numpy as np
import glob
from PIL import Image
import argparse

INP_FOLDER = 'inpaint'

parser = argparse.ArgumentParser()
parser.add_argument('--mask', type=str, default='center',
                    help='mask type, one of [left|center|random]')
args = parser.parse_args()

path_folder = os.path.join(INP_FOLDER,args.mask)
print(path_folder)


dirs = os.listdir(path_folder)
runs = [item for item in dirs if item.find('run_') > -1]

i = 0
images_inpainted = []
for run in sorted(runs):
    images = []
    images_paths = sorted(glob.glob(os.path.join(path_folder,'{}/*.png'.format(run))))
    if images_paths == []:
        raise ValueError('Folder {} is empty!'.format(os.path.join(path_folder,'{}'.format(run))))
    for file in sorted(glob.glob(os.path.join(path_folder,'{}/*.png'.format(run)))):
        image = cv2.imread(file)[13:-14,14:-13,:]
        images.append(image[:,198:2*198,:])
    images_inpainted.append(np.concatenate(images,axis = 0))
images_inpainted = np.concatenate(images_inpainted,axis = 1)

images_masked = []
images_real = []
for file in sorted(glob.glob(os.path.join(path_folder,'{}/*.png'.format(runs[0])))):
    image = cv2.imread(file)[13:-14,14:-13,:]
    images_masked.append(image[:,0:198,:])
    images_real.append(image[:,2*198:,:])
images_masked = np.concatenate(images_masked,axis = 0)
images_real = np.concatenate(images_real,axis = 0)

images_all = np.concatenate([images_masked, images_inpainted, images_real],axis = 1)

print(os.path.join(path_folder,'images_all.png'))
cv2.imwrite(os.path.join(path_folder,'images_all.png'),images_all)
