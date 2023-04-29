import os
from flask import Flask, redirect, render_template, request
from PIL import Image 
# Python Imaging Library (PIL)
# PIL is an additional, free, open-source library for the Python programming language that provides support for opening, manipulating, and saving many different image file formats.
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd








# # importing the python open cv library
# import cv2

# # intialize the webcam and pass a constant which is 0
# cam = cv2.VideoCapture(0)

# # title of the app
# cv2.namedWindow('python webcam screenshot app')

# # let's assume the number of images gotten is 0
# img_counter = 0

# # while loop
# while True:
#     # intializing the frame, ret
#     ret, frame = cam.read()
#     # if statement
#     if not ret:
#         print('failed to grab frame')
#         break
#     # the frame will show with the title of test
#     cv2.imshow('test', frame)
#     #to get continuous live video feed from my laptops webcam
#     k  = cv2.waitKey(1)
#     # if the escape key is been pressed, the app will stop
#     if k%256 == 27:
#         print('escape hit, closing the app')
#         break
#     # if the spacebar key is been pressed
#     # screenshots will be taken
#     elif k%256  == 32:
#         # the format for storing the images scrreenshotted
#         img_name = f'opencv_frame_{img_counter}'
#         # saves the image as a png file
#         cv2.imwrite(img_name, frame)
#         print('screenshot taken')
#         # the number of images automaticallly increases by 1
#         img_counter += 1

# # release the camera
# cam.release()

# # stops the camera window
# cam.destoryAllWindows()

# from cv2 import *
# import cv2 as cv
# cam_port = 0
# cam = cv.VideoCapture(cam_port)
  

# result, image = cam.read()
# if result:
  
#     # showing result, it take frame name and image 
#     # output
#     cv.imshow("GeeksForGeeks", image)
  
#     # saving image in local storage
#     cv.imwrite("GeeksForGeeks.png", image)
  
#     # If keyboard interrupt occurs, destroy image 
#     # window
#     cv.waitKey(0)
#     cv.destroyWindow("GeeksForGeeks")
  
# # If captured image is corrupted, moving to else part
# else:
#     print("No image detected. Please! try again")




disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    # print(input_data)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)

# if __name__ == '__main__':
#     app.run(debug=True)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')




@app.route('/opencvCam')
def opencv():
    return render_template('opencv.html')

@app.route('/openCapture')
def openCap():
    return render_template('openCapture.html')




@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        print(filename)
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run()
