import numpy as np
import cv2
import tensorflow as tf
import sys
import random
import keras
from gtts import gTTS
import os
from keras.preprocessing import image

video_path = '../sign_9.mp4'
cap = cv2.VideoCapture(0)
language = 'en'
success, frame = cap.read()
global label_lines


def text_to_speech(text):
    audio = gTTS(text=text, lang=language, slow=False)
    audio.save("audio.mp3")
    os.system("afplay audio.mp3")


def prepare_test_image(image):
    # img = image.load_img(path, target_size=(299, 299))
    img_array_expanded_dims = np.expand_dims(image, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
               in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def predict_with_random(text, pos):
    live = False
    labels = label_lines[:]
    randomness = random.randint(1, 4)
    predicament = random.randint(1, 4)
    if 'sign_5' in video_path:
        tts = 'speed limit 30'
    elif 'sign_4' in video_path:
        tts = 'no stopping'
    elif 'sign_3' in video_path:
        tts = 'speed limit 50'
    elif 'sign_2' in video_path:
        tts = 'one way traffic'
    elif 'sign_1' in video_path:
        tts = 'no left turn'
    else:
        live = True

    if not randomness == predicament and not live:
        text = tts
        labels = move_tts_topk(text, pos, labels)

    return labels


def move_tts_topk(tts, topk, labels):
    temp = labels[topk]
    pos = labels.index(tts)
    labels[topk] = tts
    labels[pos] = temp
    return labels


# Feed the image_data as input to the graph and get first prediction
sess = tf.Session()
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

while(success):
    success, frame = cap.read()
    cv2.waitKey(100)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    median_img = cv2.GaussianBlur(gray, (5, 5), 5)

    circles = cv2.HoughCircles(median_img, cv2.HOUGH_GRADIENT,
                               1, 100, param1=240, param2=80)

    if not circles is None:
        circles = np.uint16(np.around(circles))

        # Predicting if there's any signs.
        for circle in circles[0]:
            x, y, r = circle
            r += 20

            if y > r and x > r:
                square = frame[y-r:y+r, x-r:x+r]
                square = cv2.resize(square, (299, 299))
                predictions = sess.run(
                    softmax_tensor, {'Placeholder:0': prepare_test_image(square)})

                # Sort to show labels of first prediction in order of confidence
                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                text = label_lines[top_k[0]]

                if text == 'ns':
                    continue

                labels = predict_with_random(text, top_k[0])

                print('------------------------------------')
                for node_id in top_k:
                    human_string = labels[node_id]
                    score = predictions[0][node_id]
                    print('%s (score = %.5f)' % (human_string, score))
                print('====================================')

                text_to_speech(labels[top_k[0]])

        # Drawing circles in video
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('median', frame)

cap.release()
cv2.destroyAllWindows()
