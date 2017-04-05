#!/usr/bin/env python
# Make sure to download the ANN weights and support data with:
# $ ./data/download_data.sh

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import sys
import os.path

import numpy as np
import tensorflow as tf

from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.python.framework import ops
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import supervisor

slim = tf.contrib.slim
FLAGS = None

def PreprocessImage(image_path, central_fraction=0.875):

  if not os.path.exists(image_path):
    tf.logging.fatal('Input image does not exist %s', image_path)
  img_data = tf.gfile.FastGFile(image_path).read()

  # Decode Jpeg data and convert to float.
  img = tf.cast(tf.image.decode_jpeg(img_data, channels=3), tf.float32)

  img = tf.image.central_crop(img, central_fraction=central_fraction)
  # Make into a 4D tensor by setting a 'batch size' of 1.
  img = tf.expand_dims(img, [0])
  img = tf.image.resize_bilinear(img,
                                 [FLAGS.image_size, FLAGS.image_size],
                                 align_corners=False)

  # Center the image about 128.0 (which is done during training) and normalize.
  img = tf.multiply(img, 1.0/127.5)
  return tf.subtract(img, 1.0)


def main(args):
  if not os.path.exists(FLAGS.checkpoint):
    tf.logging.fatal(
        'Checkpoint %s does not exist. Have you download it? See data/download_data.sh',
        FLAGS.checkpoint)
  g = tf.Graph()
  with g.as_default():
    input_image = PreprocessImage(FLAGS.image_path[0])

    with slim.arg_scope(inception.inception_v3_arg_scope()):
      logits, end_points = inception.inception_v3(
          input_image, num_classes=FLAGS.num_classes, is_training=False)

    bottleneck = end_points['PreLogits']
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer(),
                       tf.tables_initializer())
    saver = tf_saver.Saver()
    sess = tf.Session()
    saver.restore(sess, FLAGS.checkpoint)

    # Run the evaluation on the image
    bottleneck_eval = np.squeeze(sess.run(bottleneck))

  first = True
  for val in bottleneck_eval:
    if not first:
      sys.stdout.write(",")
    first = False
    sys.stdout.write('{:.3f}'.format(val))
  sys.stdout.write('\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', type=str, default='data/2016_08/model.ckpt',
                      help='Checkpoint to run inference on.')
  parser.add_argument('--image_size', type=int, default=299,
                      help='Image size to run inference on.')
  parser.add_argument('--num_classes', type=int, default=6012,
                      help='Number of output classes.')
  parser.add_argument('image_path', nargs=1, default='')
  FLAGS = parser.parse_args()
  tf.app.run()