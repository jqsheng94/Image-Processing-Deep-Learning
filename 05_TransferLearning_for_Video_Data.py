import tensorflow as tf
import prettytensor as pt
import numpy as np
import os

from util import Util
u = Util()
from util_transfer import UtilTransfer
ut = UtilTransfer()

import inception

import custom_dataset as add
from custom_dataset import num_classes

add.dataset_name = "apple-drive-duck"
add.data_dir = "data/" + add.dataset_name + "/"
add.data_url = "https://github.com/Kidel/Deep-Learning-CNN-for-Image-Recognition/raw/master/customData/apple-drive-duck.tar.gz"
# Directory used for the cache files
data_dir = "cache/" + add.dataset_name +"/"
add.maybe_download_and_extract()
dataset = add.load()

class_names = dataset.class_names
class_names


image_paths_train, cls_train, labels_train = dataset.get_training_set()
image_paths_train[0]
image_paths_test, cls_test, labels_test = dataset.get_test_set()
image_paths_test[0]

print("Size of:")
print("- Training-set:\t\t{}".format(len(image_paths_train)))
print("- Test-set:\t\t{}".format(len(image_paths_test)))

def plot_images(images, cls_true, cls_pred=None, smooth=True):
    u.plot_images_2(images=images, cls_true=cls_true, class_names=class_names, cls_pred=cls_pred, smooth=smooth)

from matplotlib.image import imread

def load_images(image_paths):
    images = [imread(path) for path in image_paths]
    return np.asarray(images)

images = load_images(image_paths=(image_paths_train[3:7]+image_paths_train[555:557]+image_paths_train[-4:-1]))

cls_true = np.append(np.append(cls_train[3:7], cls_train[555:557]), cls_train[-4:-1])

plot_images(images=images, cls_true=cls_true, smooth=True)

images = load_images(image_paths=image_paths_train)
images_test = load_images(image_paths=image_paths_test)

inception.maybe_download()
model = inception.Inception()

from inception import transfer_values_cache

cache_dir = "cache/"
file_path_cache_test = os.path.join(cache_dir, 'inception_05_test.pkl')
file_path_cache_train = os.path.join(cache_dir, 'inception_05_train.pkl')

transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                             image_paths=image_paths_test,
                                             model=model)

transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              image_paths=image_paths_train,
                                              model=model)

transfer_values_train.shape
transfer_values_test.shape