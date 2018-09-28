import tensorflow as tf
from tensorflow import keras

import numpy as np
import json
import matplotlib.pyplot as plt

_n_data = np.load('imdb.npz')

#test_data = _n_data['x_test']
#train_data = _n_data['x_train']
#train_label = _n_data['y_train']
#test_label = _n_data['y_test']

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(train_data[0])

word_index = dict()
with open('imdb_word_index.json', 'r') as _file:
    word_index = json.load(_file)

word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(train_data[0]))

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
        value = word_index["<PAD>"],
        padding = 'post',
        maxlen = 256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
        value = word_index["<PAD>"],
        padding = 'post',
        maxlen = 256)

print(len(train_data[0]), len(train_data[1]))
print(train_data[0])

vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = tf.nn.relu))
model.add(keras.layers.Dense(1, activation = tf.nn.sigmoid))

print(model.summary())

model.compile(optimizer = tf.train.AdamOptimizer(),
        loss = 'binary_crossentropy',
        metrics = ['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
	partial_y_train,
	epochs = 40,
	batch_size = 512,
	validation_data = (x_val, y_val),
	verbose = 1)

history_dict = history.history

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


