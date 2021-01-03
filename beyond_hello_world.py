import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks

# model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# callbacks
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.9):
            print("\nReached 90% accuracy. Cancelling training.")
            self.model.stop_training = True

callbacks = myCallback()

# data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# normalize training values. Clamps between (0, 1)
training_images = training_images / 255.0
test_images = test_images / 255.0

# train model
model.fit(training_images, training_labels, epochs=20, callbacks=[callbacks])

# check trained model's performance
model.evaluate(test_images, test_labels)

# check classification probability
classifieds = model.predict(test_images)

# first prediction and corresponding label
print(classifieds[0])
print(test_labels[0])