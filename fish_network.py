import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os

'''trains a CNN to classify images of fish according to species'''

dir = "C:\\usrJoel\\python_Workspace\\fish_classification\\"
train_dir = os.path.join(dir, "Fish_Split_Dataset\\train")

#Load data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(256, 256),
    batch_size=64)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=64)

#Plot some images
class_names = train_ds.class_names
print(class_names)
num_classes = len(class_names)
plt.figure(figsize=(10, 10))
for i in range(num_classes):
    filtered = train_ds.filter(lambda _, l: tf.math.equal(l[0], i))
    for images, labels in filtered.take(1):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[0].numpy().astype("uint8"))
        plt.title(class_names[i])
        plt.axis("off")

plt.show()

#Define the model
model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    #layers.Dense(16, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

#Set training parameters
keras.regularizers.L2(l2=0.01)
optimizer = keras.optimizers.Adam(learning_rate=0.01)

#Compile and train the model
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

model_dir = os.path.join(dir, 'model')
model.save(model_dir)
