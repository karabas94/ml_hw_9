import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, applications, preprocessing
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""
використати transfer learning для вирішення задачі класифікації зображень
1) обрати будь яку натреновану модель з керас. використати її як backbone
2) "заморозити" ваги бекбоуна і додати кілька dense шарів для класифікації на вашому кастомному датасеті
3) натренувати модель і обчислити точність на тестовому датасеті
"""

directory = 'D:\\archive'

labels = ['paper', 'scissors', 'rock']
num_classes = len(labels)


def input_target_split(train_dir, labels):
    dataset = []
    count = 0
    for label in labels:
        folder = os.path.join(train_dir, label)
        for image in os.listdir(folder):
            img = preprocessing.image.load_img(os.path.join(folder, image), target_size=(224, 224))
            img = preprocessing.image.img_to_array(img)
            img = applications.resnet.preprocess_input(img)
            dataset.append((img, count))
        print(f'\rCompleted: {label}', end='')
        count += 1
    random.shuffle(dataset)
    X, y = zip(*dataset)

    return np.array(X), np.array(y)


X, y = input_target_split(directory, labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(np.unique(y_train, return_counts=True), np.unique(y_test, return_counts=True))

print("x_train shape:", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_train.shape[0], "test samples")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

input_shape = (224, 224, 3)

# download trained model
vgg_model = applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')

# freeze weight
vgg_model.trainable = False

# added several layers
model = keras.Sequential(
    [
        vgg_model,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ]
)

model.summary()

optimizer = keras.optimizers.Adam()

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# train model
history = model.fit(X_train, y_train, epochs=5, validation_split=0.2)

# testing model
score = model.evaluate(X_test, y_test, verbose=1)
print(f'Test loss: {score[0]}')
print(f'Test Accuracy: {score[1]}')
