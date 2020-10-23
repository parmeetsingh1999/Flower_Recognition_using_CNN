# Importing Libraries

import numpy as np
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import tensorflow as tf
import random
import cv2
from tqdm import tqdm
from random import shuffle
from PIL import Image
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from keras import backend
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Adagrad
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten, Activation, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ReduceLROnPlateau
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
style.use('fivethirtyeight')
sns.set(style = 'whitegrid', color_codes = True)

# Loading the dataset

print(os.listdir('../input/flowers-recognition/flowers'))

# Preparing train and test set from the data

x = []
y = []
IMG_SIZE = 150
tulip_dir = '../input/flowers-recognition/flowers/tulip'
daisy_dir = '../input/flowers-recognition/flowers/daisy'
sunflower_dir = '../input/flowers-recognition/flowers/sunflower'
rose_dir = '../input/flowers-recognition/flowers/rose'
dandelion_dir = '../input/flowers-recognition/flowers/dandelion'

def assign_label(img, flower_type):
    return flower_type
    
def prepare_train_data(flower_type, directory):
    for img in tqdm(os.listdir(directory)):
        label = assign_label(img, flower_type)
        path = os.path.join(directory, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        x.append(np.array(img))
        y.append(str(label))
        
prepare_train_data('Tulip', tulip_dir)
print(len(x))

prepare_train_data('Daisy', daisy_dir)
print(len(x))

prepare_train_data('Sunflower', sunflower_dir)
print(len(x))

prepare_train_data('Rose', rose_dir)
print(len(x))

prepare_train_data('Dandelion', dandelion_dir)
print(len(x))

# Visualization

fig, ax = plt.subplots(5, 2)
fig.set_size_inches(15, 15)
for i in range(5):
    for j in range(2):
        temp = random.randint(0, len(y))
        ax[i, j].imshow(x[temp])
        ax[i, j].set_title('Flower: ' + y[temp])
        
plt.tight_layout()

# Label Encoding and One Hot Encoding

le = LabelEncoder()
le_arr = le.fit_transform(y)
le_arr = to_categorical(le_arr, 5)
x = np.array(x)
x = x / 255

# Splitting of data into train and test set

x_train, x_test, y_train, y_test = train_test_split(x, le_arr, test_size = 0.25, random_state = 42)
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# CNN Model building

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'Same', activation = 'relu', input_shape = (150, 150, 3)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Conv2D(filters = 96, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Conv2D(filters = 96, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(5, activation = "softmax"))

# Applying LR Annealer

batch_size = 128
epochs = 50
red_lr = ReduceLROnPlateau(monitor = 'val_acc', patience = 3, verbose = 1, factor = 0.1)

# Data Augmentation

datagen = ImageDataGenerator(
            featurewise_center = False,
            samplewise_center = False,
            featurewise_std_normalization = False,
            samplewise_std_normalization = False,
            zca_whitening = False,
            rotation_range = 10,
            zoom_range = 0.1,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            horizontal_flip = True,
            vertical_flip = False
        )

datagen.fit(x_train)

# Compilation Phase

model.compile(optimizer = Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

# Fitting and Prediction Phase

model_fit = model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size), epochs = epochs, validation_data = (x_test, y_test), verbose = 1, steps_per_epoch = x_train.shape[0] // batch_size)

# Evaluation of Model Performance

plt.plot(model_fit.history['loss'])
plt.plot(model_fit.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(model_fit.history['accuracy'])
plt.plot(model_fit.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

# Visualizing predictions on the Validation set

pred = model.predict(x_test)
pred_digits = np.argmax(pred, axis = 1)
prop_class = []
mis_class = []
i = 0

for i in range(len(y_test)):
    if np.argmax(y_test[i]) == pred_digits[i]: prop_class.append(i)
    if len(prop_class) == 8: break

i = 0

for i in range(len(y_test)):
    if not np.argmax(y_test[i]) == pred_digits[i]: mis_class.append(i)
    if len(mis_class) == 8: break

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

count = 0
fig,ax = plt.subplots(4, 2)
fig.set_size_inches(15, 15)
for i in range(4):
    for j in range(2):
        ax[i,j].imshow(x_test[prop_class[count]])
        ax[i,j].set_title("Predicted Flower : " + str(le.inverse_transform([pred_digits[prop_class[count]]])) + "\n" + "Actual Flower : " + str(le.inverse_transform(np.argmax([y_test[prop_class[count]]]))))
        plt.tight_layout()
        count += 1

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

count = 0
fig,ax = plt.subplots(4,2)
fig.set_size_inches(15,15)
for i in range(4):
    for j in range(2):
        ax[i,j].imshow(x_test[mis_class[count]])
        ax[i,j].set_title("Predicted Flower : " + str(le.inverse_transform([pred_digits[mis_class[count]]])) + "\n" + "Actual Flower : " + str(le.inverse_transform(np.argmax([y_test[mis_class[count]]]))))
        plt.tight_layout()
        count += 1
