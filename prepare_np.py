import logging
import uuid
import cv2
import numpy as np
import argparse
import tensorflow as tf

logging.getLogger("tensorflow").setLevel(logging.ERROR)

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()

    args_parser.add_argument('--size', '-s', type=int, default=600)
    train_ratio = 1
    args = args_parser.parse_known_args()[0]
    print(args)

    task_id = uuid.uuid4()

    def crop_image(record):
        sample = tf.io.parse_single_example(record, {
            'image': tf.io.VarLenFeature(tf.int64),
            'width': tf.io.FixedLenFeature((), tf.int64),
            'height': tf.io.FixedLenFeature((), tf.int64),
        })

        image = tf.sparse_tensor_to_dense(sample['image'])
        image = tf.reshape(image, (sample['height'], sample['width'], 3))
        image = tf.image.random_crop(image, (384, 384, 3))

        return image

    # make ims as numpy format
    record = '/home/ryougoishikawa/ml-training/super-resolution/dataset/tfrecords/*.tfrecords'
    dataset = tf.data.Dataset.list_files(record)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=1, num_parallel_calls=1)
    dataset = dataset.map(crop_image)
    dataset = dataset.repeat()
    input = dataset.make_one_shot_iterator().get_next()

    ims = np.empty((0, 64, 64, 3))
    z = np.empty((0, 32, 32, 3))
    with tf.Session() as sess:
        for j in range(args.size):
            im = sess.run(input)
            cv2.imwrite('./{}.jpg'.format(j), im)
            print(j)

