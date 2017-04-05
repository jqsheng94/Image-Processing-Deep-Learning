from IPython.display import Image, display
import tensorflow as tf
import numpy as np
import os
import inception

inception.maybe_download()

model = inception.Inception()

def classify(image_path):
    display(Image(image_path))
    pred = model.classify(image_path=image_path)
    model.print_scores(pred=pred, k=10, only_first_name=True)

# Saved under inception
image_path = os.path.join(inception.data_dir, 'cat.jpg')

classify(image_path)

classify(image_path="inception/bunny3.jpg")