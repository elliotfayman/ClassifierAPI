import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import skimage
import requests
import urllib.request


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras import models
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from skimage.io import imread
from sklearn import cluster
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import sobel
from skimage.filters import sobel_h
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession




HASH = {0: ['apple_pie'], 1: ['baby_back_ribs'], 2: ['baklava'], 3: ['beef_carpaccio'], 4: ['beef_tartare'], 5: ['beet_salad'],
6: ['beignets'], 7: ['bibimbap'], 8: ['bread_pudding'], 9: ['breakfast_burrito'], 10: ['bruschetta'], 11: ['caesar_salad'],
12: ['cannoli'], 13: ['caprese_salad'], 14: ['carrot_cake'], 15: ['ceviche'], 16: ['cheese_plate'], 17: ['cheesecake'],
18: ['chicken_curry'], 19: ['chicken_quesadilla'], 20: ['chicken_wings'], 21: ['chocolate_cake'], 22: ['chocolate_mousse'],
23: ['churros'], 24: ['clam_chowder'], 25: ['club_sandwich'], 26: ['crab_cakes'], 27: ['creme_brulee'], 28: ['croque_madame'],
29: ['cup_cakes'], 30: ['deviled_eggs'], 31: ['donuts'], 32: ['dumplings'], 33: ['edamame'], 34: ['eggs_benedict'], 35: ['escargots'],
36: ['falafel'], 37: ['filet_mignon'], 38: ['fish_and_chips'], 39: ['foie_gras'], 40: ['french_fries'], 41: ['french_onion_soup'],
42: ['french_toast'], 43: ['fried_calamari'], 44: ['fried_rice'], 45: ['frozen_yogurt'], 46: ['garlic_bread'], 47: ['gnocchi'],
48: ['greek_salad'], 49: ['grilled_cheese_sandwich'], 50: ['grilled_salmon'], 51: ['guacamole'], 52: ['gyoza'], 53: ['hamburger'],
54: ['hot_and_sour_soup'], 55: ['hot_dog'], 56: ['huevos_rancheros'], 57: ['hummus'], 58: ['ice_cream'], 59: ['lasagna'],
60: ['lobster_bisque'], 61: ['lobster_roll_sandwich'], 62: ['macaroni_and_cheese'], 63: ['macarons'], 64: ['miso_soup'], 65: ['mussels'],
66: ['nachos'], 67: ['omelette'], 68: ['onion_rings'], 69: ['oysters'], 70: ['pad_thai'], 71: ['paella'], 72: ['pancakes'],
73: ['panna_cotta'], 74: ['peking_duck'], 75: ['pho'], 76: ['pizza'], 77: ['pork_chop'], 78: ['poutine'], 79: ['prime_rib'],
80: ['pulled_pork_sandwich'], 81: ['ramen'], 82: ['ravioli'], 83: ['red_velvet_cake'], 84: ['risotto'], 85: ['samosa'], 86: ['sashimi'],
87: ['scallops'], 88: ['seaweed_salad'], 89: ['shrimp_and_grits'], 90: ['spaghetti_bolognese'], 91: ['spaghetti_carbonara'],
92: ['spring_rolls'], 93: ['steak'], 94: ['strawberry_shortcake'], 95: ['sushi'], 96: ['tacos'], 97: ['takoyaki'],
98: ['tiramisu'], 99: ['tuna_tartare'], 100: ['waffles']}

def predict_image(filename,model):
    try:
        get = requests.get(filename)
        if get.status_code == 200:
            with urllib.request.urlopen(filename) as url:
                with open('temp.jpg', 'wb') as f:
                    f.write(url.read())
                filename = 'temp.jpg'

        else:
            raise requests.exceptions.RequestException
    except requests.exceptions.RequestException as e:
        return "Image URL invalid"

    img_ = image.load_img(filename, target_size=(299, 299))

    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.

    prediction = model.predict(img_processed)
    #Delete temp.jpg if it exists
    if os.path.exists('temp.jpg'):
        os.remove('temp.jpg')
    return np.argmax(prediction)

def main(image_url):

    K.clear_session()


    model = load_model('/home/elliotfayman/mysite/model_v1_inceptionV3.h5')

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    return HASH[predict_image(image_url, model)][0]
def getHash():
    return HASH
