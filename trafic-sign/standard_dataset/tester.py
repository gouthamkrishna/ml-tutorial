import numpy as np
import cv2
import tensorflow as tf
import os
import sys
import keras
from keras.preprocessing import image

def prepare_test_image(image):
    img_array_expanded_dims = np.expand_dims(image, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
               in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists first graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

sess = tf.Session()
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

for filename in os.listdir('noovertaketest'):
    
    print('------------------------------------')
    filename = 'noovertaketest/' + filename
    print ('File : ', filename)

    img = image.load_img(filename, target_size=(299, 299))
    img_array = image.img_to_array(img)
    
    predictions = sess.run(
         softmax_tensor, {'Placeholder:0': prepare_test_image(img_array)})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    text = label_lines[top_k[0]]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
    print('====================================')

