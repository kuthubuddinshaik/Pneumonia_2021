import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, layers
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


import os
import numpy as np
import pandas as np
import seaborn as sns

from random import randint
from PIL import Image
from keras.models import load_model

from flask import Flask,render_template,request
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

model = load_model('models/model.h5')

COUNT = 0

def predict_image(filename):
    img_height,img_width = 180, 180
    img = load_img(filename, target_size=(img_height, img_width))
    image = keras.preprocessing.image.img_to_array(img)
    image = image / 255.0
    image = image.reshape(1,180,180,3) 
    prediction = model.predict(image)
    if(prediction[0] > 0.5):
        stat = prediction[0] * 100 
        return("This image is %.2f percent %s"% (stat, "PNEUMONIA"))
    else:
        stat = (1.0 - prediction[0]) * 100
        return("This image is %.2f percent %s" % (stat, "NORMAL"))


@app.route("/",methods=["GET","POST"])
def home():
    if request.method == "POST":
        global COUNT
        COUNT+=1
        img = request.files['image']
        file_name="{}.jpg".format(COUNT)
        basepath=os.path.dirname(__file__)
        file_path = os.path.join(basepath,'uploads',secure_filename(file_name))
        img.save(file_path)   
        msg = predict_image(file_path)
        return msg
    else:
        return render_template("index.html")

if(__name__=='__main__'):
    app.run()    






