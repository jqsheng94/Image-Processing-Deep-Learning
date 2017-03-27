from IPython.display import Image, display
import tensorflow as tf
import numpy as np
import os
from util import Util
u = Util()
from util_transfer import UtilTransfer
ut = UtilTransfer()
import inception


inception.maybe_download()

model = inception.Inception()

def classify(image_path):
    display(Image(image_path))
    pred = model.classify(image_path=image_path)
    model.print_scores(pred=pred, k=10, only_first_name=True)

import prettytensor as pt

import cifar10
from cifar10 import num_classes
# cifar10.data_path = "data/CIFAR-10/"

cifar10.maybe_download_and_extract()

class_names = cifar10.load_class_names()
class_names



images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()


n1 = 100
n2 = 60
images_train = images_train[:n1]
cls_train = cls_train[:n1]
labels_train = labels_train[:n1]

images_test = images_test[:n2]
cls_test = cls_test[:n2]
labels_test = labels_test[:n2]

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))

def plot_images(images, cls_true, cls_pred=None, smooth=True):
    u.plot_images_2(images=images, cls_true=cls_true, class_names=class_names, cls_pred=cls_pred, smooth=smooth)


images = images_test[50:59]
cls_true = cls_test[50:59]
plot_images(images=images, cls_true=cls_true, smooth=False)


inception.data_dir = 'inception/'
cache_dir = "cache/"
inception.maybe_download()

model = inception.Inception()

from inception import transfer_values_cache

file_path_cache_train = os.path.join(cache_dir, 'inception_cifar10_train.pkl')
file_path_cache_test = os.path.join(cache_dir, 'inception_cifar10_test.pkl')

images_scaled = images_train * 255.0

transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              images=images_scaled,
                                              model=model)

images_scaled = images_test * 255.0

transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                             images=images_scaled,
                                             model=model)

transfer_values_train.shape

transfer_values_test.shape

def plot_transfer_values(i):
    u.plot_transfer_values(i=i, images=images_test, transfer_values=transfer_values_test)

plot_transfer_values(190)


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

transfer_values = transfer_values_train[0:1000]
cls = cls_train[0:1000]

# First we reduce the dimensions from 2048 to 50 using PCA
pca = PCA(n_components=50)
transfer_values_50d = pca.fit_transform(transfer_values)

# Then we reduce to 2 dimensions using t-SNE
tsne = TSNE(n_components=2)
transfer_values_reduced = tsne.fit_transform(transfer_values_50d)

# Helper function to plot the scatter
def plot_scatter(values, cls):
    u.plot_scatter(values=values, cls=cls, num_classes=num_classes)

plot_scatter(transfer_values_reduced, cls)


transfer_len = model.transfer_len
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

x_pretty = pt.wrap(x)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        fully_connected(size=1024, name='layer_fc1').\
        softmax_classifier(class_count=num_classes, labels=y_true)

global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)
# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)

y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
# The classification accuracy is calculated by first type-casting the array of booleans to floats,
# so that False becomes 0 and True becomes 1, and then taking the average of these numbers.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.initialize_all_variables())

train_batch_size = 64

def random_batch():
    ut.random_batch(transfer_values_train, train_batch_size, labels_train)

def optimize(num_iterations):
    ut.optimize(num_iterations, transfer_values_train, train_batch_size,
                labels_train, session, global_step, optimizer, accuracy, x, y_true)


def plot_example_errors(cls_pred, correct):
    ut.plot_example_errors(cls_pred, correct, images_test, cls_test, plot_images, images)

def plot_confusion_matrix(cls_pred):
    ut.plot_confusion_matrix(cls_pred, cls_test, num_classes, class_names)

batch_size = 256

def predict_cls(transfer_values, labels, cls_true):
    # Number of images.
    num_images = len(transfer_values)
    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    # Now calculate the predicted classes for the batches.
    i = 0
    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)
        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: transfer_values[i:j],
                     y_true: labels[i:j]}
        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        # Set the start-index for the next batch to the end-index of the current batch.
        i = j
    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)
    return correct, cls_pred

def classification_accuracy(correct):
    ut.classification_accuracy(correct)

def print_test_accuracy(show_example_errors=False, show_confusion_matrix=False):
    ut.print_test_accuracy(show_example_errors, show_confusion_matrix, transfer_values_test,
                           labels_test, cls_test, batch_size, images_test, plot_images, images,
                           num_classes, class_names, predict_cls)


print_test_accuracy(show_example_errors=False, show_confusion_matrix=True)

optimize(num_iterations=1000)

print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)

model.close()
session.close()



