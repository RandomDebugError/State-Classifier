# Deep Learning Fundamentals course
import numpy as np
import os
import pathlib
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# Default parameters, modify and find what's best for your model
batch_size = 128
img_size = 256
learning_rate = 0.0004
num_epochs = 100
dropout = 0.5
patience = 15

#imagegenerator to increase and augment dataset provided. Any added augmentations only degrade accuracy.
train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        shear_range=0.3,
        zoom_range=0.2,
        rotation_range=15,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True)



#new function for importing files from training dierectory
train_generator = train_datagen.flow_from_directory(
        'data/train',  
        target_size=(img_size, img_size),
        batch_size=64,
        seed = 3,
        shuffle=True)     



def load_dataset(ds, mode):

  def process_path(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    label = tf.cast(one_hot, dtype='int64')  # Create labels
    img = tf.io.read_file(file_path)  # Read file
    img = tf.io.decode_jpeg(img, channels=3)  # Convert image to tensor
    img = tf.image.convert_image_dtype(img, tf.float32) # Normalize image in the range of 0 and 1
    img = tf.image.resize(img, [img_size, img_size])    # Resize the image
    return img, label

  def transformation(image, label):
    # Apply transformations to images 
    # Removed if option for training data as it is now handled by image generator
      img = tf.clip_by_value(image, 0.0,1.0) 
      return img, label

  def configuration(dataset):
    # Configurations to optimize the performance
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.map(transformation, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
  
  # Load and generate dataset
  data_dir = pathlib.Path(ds) # Specify the path to data folder
  image_count = len(list(data_dir.glob('*/*.jpg')))
  files_list = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
  files_list = files_list.shuffle(image_count, reshuffle_each_iteration=False)
  class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
  print("Loading {} dataset!".format(ds))
  print("Number of classes: {}.".format(len(class_names)))
  print("Data size: {}.".format(image_count))
  print(" ---------- --------- ---------- \n")
  return configuration(files_list)

# Prepare train and valid dataset, removed loading of training dataset
valid_ds = load_dataset(ds = 'data/valid', mode=None)

num_classes = 11

vgg16_model = tf.keras.applications.vgg16.VGG16(include_top = False, weights = 'imagenet', pooling = 'maxpooling', input_shape = (img_size, img_size, 3))
vgg16_model_touse = tf.keras.models.Model(inputs = vgg16_model.input, outputs= vgg16_model.output)

model = tf.keras.Sequential()
model.add(vgg16_model_touse)


model.add(tf.keras.layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), padding = 'same', activation = 'relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size = (2, 2)))
model.add(tf.keras.layers.GlobalMaxPool2D())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))


model.summary()
model.layers[0].trainable = False

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-6),
    loss = 'categorical_crossentropy',
    metrics=['accuracy'])

# Make sure to use h5 format to save model

callbacks = [EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min'), 
            ModelCheckpoint(filepath='model.h5', verbose=1, monitor='val_loss', save_best_only=True, save_weights_only=False, mode='min')]

model.fit(
  train_generator,
  epochs=num_epochs,
  validation_data=(valid_ds),
  callbacks=callbacks
)