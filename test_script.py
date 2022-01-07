import numpy as np
import os
import pathlib
import tensorflow as tf
import json

batch_size = 128
img_size = 256

# Set class names manually
# d = {0: 'creamy_paste',
#  1: 'diced',
#  2: 'floured',
#  3: 'grated',
#  4: 'juiced',
#  5: 'jullienne',
#  6: 'mixed',
#  7: 'other',
#  8: 'peeled',
#  9: 'sliced',
#  10: 'whole'}

# Get class names and index in a dictionary
data_dir = pathlib.Path("data/train") # Specify the path to train data folder
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
d = {v: k for v, k in enumerate(class_names)}

# Load test images
data_dir = pathlib.Path("data/test/test") # Specify the path to test data folder
image_count = len(list(data_dir.glob('*.jpg')))
files_list = tf.data.Dataset.list_files(str(data_dir/'*'), shuffle=False)
print("Loading test dataset \nData size: {}".format(image_count))

def process_path(file_path):
  img = tf.io.read_file(file_path)  # Read file
  img = tf.io.decode_jpeg(img, channels=3)  # Convert image to tensor
  img = tf.image.convert_image_dtype(img, tf.float32) # Normalize image in the range of 0 and 1
  img = tf.image.resize(img, [img_size, img_size])    # Resize the image
  return img

def transformation(image):
  # Apply transformations to images
  # the type of transformation (augmentations) used is different at train and evaluation time.

  img = tf.clip_by_value(image, 0.0,1.0)    # To make sure images are in the range of 0 and 1 after transformations
  return img

# Process test images
dataset = files_list.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.map(transformation, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(batch_size=batch_size)

# Load saved model
new_model = tf.keras.models.load_model("model.h5")

# Predict
pred = new_model.predict(dataset, verbose=1)
preds = np.argmax(pred, axis=1)

# Create dictionary of images and their predictions
dictionary = {}
count = 0
for i in files_list:
  path = i.numpy().decode('utf-8')
  #edited dictionary items to make them proper for comparison in evaluate.py
  dictionary.update({"test_anonymous/" + path[15:]:d[preds[count]]})
  count += 1
  #added so thsat it only looks at first 1000, since thats what was asked
  if count == 1000:
    break

with open("results.json", "w") as outfile:
  json.dump(dictionary, outfile) 

print("Json file saved.")