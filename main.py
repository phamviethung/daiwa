from keras import applications
from keras import optimizers
from keras.layers import Dropout
from keras import models
from keras import layers
import serial
import thread
import tensorflow as tf
import pygame
import pygame.camera
import os
import cv2

def cnn_model():
    img_width, img_height = 320, 240

    # build the VGG16 network
    conv_base = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    conv_base.trainable = True
    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == 'block3_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(4, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # load weights
    model.load_weights('Fine-tuning-daiwa-30-conv3-data8.h5')
    print('Model loaded.')
    return model

def predict(model, data):
    from keras.preprocessing import image
    import numpy as np


    img = image.load_img(data, target_size=(240, 320))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    '''
    img = cv2.imread(data)
    img_tensor = np.expand_dims(img, axis=0)
    img_tensor /= 255.
    '''
    out_1 = model.predict(img_tensor)
    out_2 = model.predict_classes(img_tensor)
    return out_1, out_2


os.system('v4l2-ctl -d 0 -c focus_absolute=40')
os.system('v4l2-ctl -d 0 -c brightness=85')

DEVICE = '/dev/video0'
SIZE = (640, 480)

model = cnn_model()
pygame.init()
pygame.camera.init()
display = pygame.display.set_mode(SIZE, 0)
camera = pygame.camera.Camera(DEVICE, SIZE)
camera.start()
screen = pygame.surface.Surface(SIZE, 0, display)

ser = serial.Serial('/dev/ttyUSB0', 115200)
state = 1

graph = tf.get_default_graph()


def get_data_sensor(threadName, delay):
    while True:
        global state
        state = ser.readline()

def cam_run(threadName, delay):
    index = 0
    while True:
        #time.sleep(delay)
        global screen
        screen = camera.get_image(screen)
        display.blit(screen, (0, 0))
        pygame.display.flip()
        pygame.display.update()
        global state
        if int(state) == 0:
            pygame.image.save(screen, 'thanh' + str(index) + '.jpg')
            state = 1
            global graph
            with graph.as_default():
                path = 'thanh' + str(index) + '.jpg'
                print path
		img = cv2.imread(path, 0)
    		ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)
    		median = cv2.medianBlur(thresh, 3)
    		cv2.imwrite(path, median)
                out_1, out_2 = predict(model, path)
                print out_1
                if out_2[0] == 0:
                    print 'NG'
                elif out_2[0] == 1:
                    print 'OK'
                elif out_2[0] == 2:
                    print 'COLOR'
                else:
                    print 'BARI'

            index += 1


# Create two threads as follows
try:
   thread.start_new_thread(get_data_sensor, ("Thread-1", 0))
   thread.start_new_thread(cam_run, ("Thread-2", 0))
except:
   print "Error: unable to start thread"

while 1:
   pass

