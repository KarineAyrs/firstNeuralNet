import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# определяет, нарисован + или -
data = np.array([[0, 1, 0, 1, 1, 1, 0, 1, 0],  # +
                 [0, 1, 0, 1, 1, 1, 0, 1, 0],  # +
                 [0, 0, 0, 1, 1, 1, 0, 0, 0],  # -
                 [0, 0, 0, 1, 1, 1, 0, 0, 0],  # -
                 [0, 1, 0, 1, 1, 1, 0, 1, 0],  # +
                 [0, 0, 0, 1, 1, 1, 0, 0, 0],  # -
                 [0, 0, 0, 1, 1, 1, 0, 0, 0],  # -
                 [0, 1, 0, 1, 1, 1, 0, 1, 0],  # +
                 ])
label = np.array([1, 1, 0, 0, 1, 0, 0, 1])
train_data = data[:4]
train_label = label[:4]

test_data = data[4:]
test_label = label[4:]

model = keras.Sequential()
model.add(keras.layers.Dense(16, input_shape=[9]))
model.add(keras.layers.Dense(64, activation=tf.nn.sigmoid))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='mse',
              metrics=['accuracy'])

model.summary()
history = model.fit(train_data,
                    train_label,
                    batch_size=2,
                    epochs=100,
                    validation_data=(test_data, test_label),
                    verbose=1)

test_loss, test_acc = model.evaluate(test_data, test_label)
print('\nТочность на проверочных данных', test_acc)
predictions = model.predict(test_data)
print(predictions)
history_dict = history.history
history_dict.keys()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" означает "blue dot", синяя точка
plt.plot(epochs, loss, 'bo', label='Потери обучения')
# "b" означает "solid blue line", непрерывная синяя линия
plt.plot(epochs, val_loss, 'b', label='Потери проверки')
plt.title('Потери во время обучения и проверки')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()

plt.show()

plt.clf()  # Очистим график
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Точность обучения')
plt.plot(epochs, val_acc, 'b', label='Точность проверки')
plt.title('Точность во время обучения и проверки')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()
