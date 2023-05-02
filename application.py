from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import cv2
import numpy as np
import os
app = Flask(__name__)
@app.route('/')
def home():
   return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        file.filename="image.JPG"
        file_path = os.path.join('static/images',"image.JPG")
        file.save(file_path)
        model=load_model('potato_model.h5',custom_objects={'KerasLayer':hub.KerasLayer})
        img_potato=cv2.imread('static/images/image.JPG')
        img_potato_resized=cv2.resize(img_potato,(224,224))
        img_potato_scaled=img_potato_resized/255
        img_potato_cl=img_potato_scaled[np.newaxis, ...]
        pred=model.predict(img_potato_cl)
        result=np.argmax(pred)
        if result==0:
             res='Healthy Potato'
        elif result==1:
             res='Unhealthy Potato: Early Blight'
        else:
            res='Unhealthy Potato: Late Blight'
        return render_template('predict.html',user_image=file_path,classif_res=res)
if __name__ == '__main__':
   app.run(debug = True)
