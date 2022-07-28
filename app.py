from cv2 import COLOR_BGR2GRAY
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from keras_preprocessing import image 
from flask import request, jsonify
from flask import Flask, render_template, url_for
import tensorflow as tf

import random



app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
MODEL_PATH = 'PolypCModel4.h5'
MODEL_PATH2 = 'bestyet.h5'

def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + 1)

def jacard_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection + 1.0)

model = load_model(MODEL_PATH)
model.make_predict_function()      


model2 = load_model(MODEL_PATH2, custom_objects={'dice_coef':dice_coef, 'jacard_coef':jacard_coef})
model2.make_predict_function()  


@app.route('/', methods=['GET'])

def index():
    #main page
    return render_template('login.html')
    
@app.route('/index', methods=['GET'])

def login():
    return render_template('index.html')

@app.route('/',methods=['GET','POST'])

def predict():
    imgfile = request.files['imgfile']
    image_path = "./static/uploads/" + imgfile.filename
    mask_path = "./static/mask/mask"+str(random.randint(0,100))+".jpg"
    imgfile.save(image_path)

    # img = image.load_img(image_path, target_size=(224, 224))
    # img = image.img_to_array(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = np.expand_dims(img, axis=0)
    # img = cv2.imread(image_path)
    # img = cv2.resize(img, (224, 224), COLOR_BGR2GRAY)   
    # img = image.img_to_array(img)
    # img = np.array(img)
    # img = np.expand_dims(img, axis=0)

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128),color_mode='rgb')
    img = np.asarray(img)
    img  = np.expand_dims(img,axis=0)
    img=img/255.0
    class_preds = model.predict(img)
    class_preds = int((class_preds[0][0]).round())
    


    if class_preds ==0:
        ans = 'Not Polyp'
    else:
        ans = 'Polyp'
        seg_preds = model2.predict(img)
        seg_preds = (seg_preds > 0.2).astype(np.uint8)
        seg_preds = seg_preds[0]
        seg_preds = cv2.cvtColor(seg_preds,cv2.COLOR_GRAY2BGR)
        # seg_preds  = np.expand_dims(seg_preds,axis=0)
        img = img[0]
        # img = img.astype(np.float) 
        img=img*255.0 
        # img=img*255.0 
        seg_preds=seg_preds*255.0
        print(img.shape)
        print(seg_preds.shape)
        added_image = cv2.addWeighted(img,1,seg_preds,0.3,0)

        # cv2.imwrite(mask_path,added_image*255.0)
        # cv2.imwrite(mask_path,img)
        # cv2.imwrite("./static/mask/mask.jpg",img)
        tf.keras.utils.save_img(mask_path, added_image)





    return render_template('result.html', prediction_text=ans, image=mask_path)

@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

# @app.route('/result')

# def result():
#     return render_template('result.html')


if __name__ == "__main__":
    app.run(debug=True)
