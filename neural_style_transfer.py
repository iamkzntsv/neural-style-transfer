# Imports
import os
import sys
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import urllib
from PIL import Image
from io import BytesIO
import sklearn.metrics
from model.vgg import *

# Define layer for computing content cost and its weight
CONTENT_LAYER = [('block5_conv4', 1)]

# Define layers and their weights for computing style cost
STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]

# Build the model
model = NST().build(STYLE_LAYERS + CONTENT_LAYER)

# PREPROCESSING
preprocessed_content = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C = vgg_model_outputs(preprocessed_content)

preprocessed_style = tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S = vgg_model_outputs(preprocessed_style)