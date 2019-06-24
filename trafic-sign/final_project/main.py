import numpy as np
import cv2
import tensorflow as tf
import sys
import keras
from keras.preprocessing import image

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('../sign_5.mp4')
sign_count = 0
no_sign_count = 0
success, frame = cap.read()


def prepare_test_image(image):
    # img = image.load_img(path, target_size=(299, 299))
    # img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(image, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


# Loads label file, strips off carriage return
primary_label_lines = [line.rstrip() for line
               in tf.gfile.GFile("primary_labels.txt")]
secondary_label_lines = [line.rstrip() for line
               in tf.gfile.GFile("secondary_labels.txt")]

# Unpersists first graph from file
with tf.gfile.FastGFile("primary_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

primary_sess = tf.Session()
primary_softmax_tensor = primary_sess.graph.get_tensor_by_name('final_result:0')

# Unpersists second graph from file
with tf.gfile.FastGFile("secondary_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

secondary_sess = tf.Session()
secondary_softmax_tensor = secondary_sess.graph.get_tensor_by_name('final_result:0')

while(cap.isOpened()):
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
                primary_predictions = primary_sess.run(
                    primary_softmax_tensor, {'Placeholder:0': prepare_test_image(square)})

                # Sort to show labels of first prediction in order of confidence
                top_k = primary_predictions[0].argsort()[-len(primary_predictions[0]):][::-1]
                primary_text = primary_label_lines[top_k[0]]

                if primary_text == '0':
                    print('not a sign')
                    continue

                secondary_predictions = secondary_sess.run(
                    secondary_softmax_tensor, {'Placeholder:0': prepare_test_image(square)})

                # Sort to show labels of first prediction in order of confidence
                top_k = secondary_predictions[0].argsort()[-len(secondary_predictions[0]):][::-1]
                secondary_text = secondary_label_lines[top_k[0]]

                print(secondary_text)
                # print('------------------------------------')
                # for node_id in top_k:
                #     human_string = label_lines[node_id]
                #     score = primary_predictions[0][node_id]
                #     print('%s (score = %.5f)' % (human_string, score))
                # print('====================================')

        # Drawing circles in video
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('test video', frame)

cap.release()
cv2.destroyAllWindows()
