import matplotlib.pyplot as plt
import tensorflow as tf


# model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# normalize training values. Clamps between to (0, 1)
training_images = training_images / 255.0
test_images = test_images / 255.0

# train model
model.fit(training_images, training_labels, epochs=20)

# check trained model's performance
model.evaluate(test_images, test_labels)
