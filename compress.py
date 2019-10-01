from PIL import Image
import glob
from scipy.misc import imresize
import numpy as np
from numpy import array
from numpy.random import randint
import cv2
import os

def lr_images(images_real, downscale, quality):
    def imgEncodeDecode(img, quality=20):
        # quality = random.randint(15, 40)  # Decide quality factor(15-40) by random
        # quality = 20
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

        result, encimg = cv2.imencode('.jpg', img, encode_param)
        if False == result:
            print('could not encode image!')
            exit()

        decimg = cv2.imdecode(encimg, 3)

        return decimg

    images = []
    for img in range(len(images_real)):
        images.append(imgEncodeDecode(
            imresize(images_real[img], [images_real[img].shape[0] // downscale, images_real[img].shape[1] // downscale],
                     interp='bicubic', mode=None), quality))
    images_lr = array(images)
    return images_lr


l = sorted(list(glob.glob('/home/rishikawa/Flask_keras/test/*.jpg')))

q = 10
os.makedirs('/home/rishikawa/Flask_keras/q_' + str(q), exist_ok=True)

for i in range(len(l)):
    img = cv2.imread(l[i])
    img = cv2.resize(img, (350, 350))
    img = lr_images([img], 1, quality=q)
    # img = lr_images([img], 2, quality=q)
    print(img.shape)
    cv2.imwrite('/home/rishikawa/Flask_keras/q_' + str(q) + '/' + str(i) + '.png', img[0])
