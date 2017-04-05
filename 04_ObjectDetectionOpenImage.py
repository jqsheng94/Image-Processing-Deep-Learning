#!/usr/bin/env python
# Make sure to download the ANN weights and support data with:
# $ ./data/download_data.sh

#source activate tensorflow
#go to folder ~Image-Processing-Deep-Learning
#Run the script in the command line
# python data/classify.py inception/starrynight.jpg

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

def PreprocessImage(image, central_fraction=0.875):
  image = tf.cast(tf.image.decode_jpeg(image, channels=3), tf.float32)

  image = tf.image.central_crop(image, central_fraction=central_fraction)

  image = tf.expand_dims(image, [0])
  image = tf.image.resize_bilinear(image,
                                 [FLAGS.image_size, FLAGS.image_size],
                                 align_corners=False)

  # Center the image about 128.0 (which is done during training) and normalize.
  image = tf.multiply(image, 1.0/127.5)
  return tf.subtract(image, 1.0)


def LoadLabelMaps(num_classes, labelmap_path, dict_path):

  labelmap = [line.rstrip() for line in tf.gfile.GFile(labelmap_path).readlines()]
  if len(labelmap) != num_classes:
    tf.logging.fatal(
        "Label map loaded from {} contains {} lines while the number of classes is {}".format(
            labelmap_path, len(labelmap), num_classes))
    sys.exit(1)

  label_dict = {}
  for line in tf.gfile.GFile(dict_path).readlines():
    words = [word.strip(' "\n') for word in line.split(',', 1)]
    label_dict[words[0]] = words[1]

  return labelmap, label_dict


def main(args):
  if not os.path.exists(FLAGS.checkpoint):
    tf.logging.fatal(
        'Checkpoint %s does not exist. Have you download it? See data/download_data.sh',
        FLAGS.checkpoint)
  g = tf.Graph()
  with g.as_default():
    input_image = tf.placeholder(tf.string)
    processed_image = PreprocessImage(input_image)

    with slim.arg_scope(inception.inception_v3_arg_scope()):
      logits, end_points = inception.inception_v3(
          processed_image, num_classes=FLAGS.num_classes, is_training=False)

    predictions = end_points['multi_predictions'] = tf.nn.sigmoid(
        logits, name='multi_predictions')
    saver = tf_saver.Saver()
    sess = tf.Session()
    saver.restore(sess, FLAGS.checkpoint)

    # Run the evaluation on the images
    for image_path in FLAGS.image_path:
      if not os.path.exists(image_path):
        tf.logging.fatal('Input image does not exist %s', FLAGS.image_path[0])
      img_data = tf.gfile.FastGFile(image_path).read()
      print(image_path)
      predictions_eval = np.squeeze(sess.run(predictions,
                                             {input_image: img_data}))

      # Print top(n) results
      labelmap, label_dict = LoadLabelMaps(FLAGS.num_classes, FLAGS.labelmap, FLAGS.dict)

      top_k = predictions_eval.argsort()[-FLAGS.n:][::-1]
      for idx in top_k:
        mid = labelmap[idx]
        display_name = label_dict.get(mid, 'unknown')
        score = predictions_eval[idx]
        print('{}: {} - {} (score = {:.2f})'.format(idx, mid, display_name, score))
      print()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', type=str, default='data/2016_08/model.ckpt',
                      help='Checkpoint to run inference on.')
  parser.add_argument('--labelmap', type=str, default='data/2016_08/labelmap.txt',
                      help='Label map that translates from index to mid.')
  parser.add_argument('--dict', type=str, default='dict.csv',
                      help='Path to a dict.csv that translates from mid to a display name.')
  parser.add_argument('--image_size', type=int, default=299,
                      help='Image size to run inference on.')
  parser.add_argument('--num_classes', type=int, default=6012,
                      help='Number of output classes.')
  parser.add_argument('--n', type=int, default=10,
                      help='Number of top predictions to print.')
  parser.add_argument('image_path', nargs='+', default='')
  FLAGS = parser.parse_args()
  tf.app.run()
