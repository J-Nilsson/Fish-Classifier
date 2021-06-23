import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import precision_score, classification_report, \
    recall_score, accuracy_score, f1_score
import os

'''evalutes a CNN trained to classify images of fish according to species'''

dir = "C:\\usrJoel\\python_Workspace\\fish_classification\\"
test_dir = os.path.join(dir, "Fish_Split_Dataset\\test")

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  image_size=(256, 256),
  shuffle=False)

model_dir = os.path.join(dir, "model")
model = keras.models.load_model(model_dir)
model.summary()

predictions = np.array([])
labels =  np.array([])
for x, y in test_ds:
  predictions = np.concatenate([predictions, model.predict_classes(x)])
  labels = np.concatenate([labels, y.numpy()])

matrix = tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()
fish_labels = ['Forell', 'Förgylld braxen', 'Havsaborre', 'Makrill', 'Randig multe', 'Räka', 'Röd braxen', 'Röd multe', 'Skarpsill']
sn.heatmap(matrix, annot=True, annot_kws={"size": 16}, xticklabels=fish_labels, yticklabels=fish_labels)
plt.show()

print(classification_report(labels, predictions))
print('Accuracy:', accuracy_score(labels, predictions))
print('F1 score:', f1_score(labels, predictions, average="macro"))
print('Recall:', recall_score(labels, predictions, average="macro"))
print('Precision:', precision_score(labels, predictions, average="macro"))
