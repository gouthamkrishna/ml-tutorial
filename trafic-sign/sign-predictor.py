from __future__ import absolute_import, division, print_function

import tensorflow as tf
import pathlib
import random
import matplotlib.pyplot as plt
import keras
from keras.preprocessing import image
import numpy as np

tf.enable_eager_execution()

print("tf version: ", tf.VERSION)

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32


def caption_image(image_path):
    return "Sample Image"


def change_range(image, label):
    return 2*image-1, label


def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [192, 192])
    image = tf.cast(image, tf.float32)
    image = image/255.0  # normalize to [0,1] range

    return image


def prepare_test_image(path):
    img = image.load_img(path, target_size=(192, 192))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


data_root = pathlib.Path('circles/')

print("Directories: ")
for item in data_root.iterdir():
    print('\t', item)

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print("Image count: ", image_count)
print("Sample image paths: ", all_image_paths[:5])

label_names = sorted(
    item.name for item in data_root.glob('*/') if item.is_dir())

label_to_index = dict((name, index) for index, name in enumerate(label_names))
print("Label to index mapping: ", label_to_index)

all_image_labels = [label_to_index[pathlib.Path(
    path).parent.name]for path in all_image_paths]

# Loading a sample image
img_path = all_image_paths[0]
label = all_image_labels[0]
img_raw = tf.read_file(img_path)
img_tensor = tf.image.decode_image(img_raw)
print("Sample image shape: ", img_tensor.shape)
print("Sample image type: ", img_tensor.dtype)

image_path = all_image_paths[0]
label = all_image_labels[0]

# Displaying a sample image
plt.imshow(load_and_preprocess_image(img_path))
plt.grid(False)
plt.xlabel(caption_image(img_path))
plt.title(label_names[label].title())
plt.show()

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# print('Path DS shape: ', repr(path_ds.output_shapes))
# print('Path DS type: ', path_ds.output_types)
# print('Path DS: ', path_ds)

image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(
    tf.cast(all_image_labels, tf.int64))

# image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
# print('Image DS shape: ', image_label_ds.output_shapes[0])
# print('Label DS shape: ', image_label_ds.output_shapes[1])
# print('Image Label DS: ', image_label_ds)

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
image_label_ds = ds.map(load_and_preprocess_from_path_label)

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
ds = ds.prefetch(buffer_size=AUTOTUNE)
print('DataSet to train: ', ds)

mobile_net = tf.keras.applications.MobileNetV2(
    input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable = False

keras_ds = ds.map(change_range)
image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(label_names))])

logit_batch = model(image_batch).numpy()
print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print("Shape:", logit_batch.shape)

model.compile(optimizer=tf.train.AdamOptimizer(
), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=["accuracy"])

steps_per_epoch = tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
print("Steps per epoch: ", steps_per_epoch)

model.fit(ds, epochs=3, steps_per_epoch=34)

test_image = prepare_test_image(image_path)
predictions = model.predict(test_image)
print(predictions[0])
