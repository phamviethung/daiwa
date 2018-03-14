from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras import models
from keras import layers

# path to the model weights files.
#weights_path = '../keras/examples/vgg16_weights.h5'
#top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 320, 240
batch_size = 32

train_data_dir = '/home/hungpv/data8/train'
validation_data_dir = '/home/hungpv/data8/validation'


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
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
#top_model = Sequential()
#top_model.add(Flatten(input_shape=conv_base.output_shape[1:]))
#top_model.add(Dense(256, activation='relu'))
#top_model.add(Dropout(0.5))
#top_model.add(Dense(1, activation='sigmoid'))


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(4, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
#top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
# model.add(top_model) # Their approach would work if we weren't trying to reload the top model weights from part 2 of the tutorial

# model = Model(input= base_model.input, output= top_model(base_model.output))

# set the first 15 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
#for layer in model.layers[:15]:
    #layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    channel_shift_range=30,
    #rotation_range=10,
    #width_shift_range=0.1,
    #height_shift_range=0.1,
    #shear_range=0.1,
    #zoom_range=0.1,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)

#model.save_weights('Fine-tuning-daiwa-30-conv3-data8.h5')