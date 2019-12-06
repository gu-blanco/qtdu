from keras.layers import Input, Concatenate, concatenate, Merge, Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.metrics import top_k_categorical_accuracy
from keras.models import Sequential, Model, load_model
import tensorflow as tf
from keras.initializers import he_uniform
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, CSVLogger, ReduceLROnPlateau
import keras.backend.tensorflow_backend as KTF
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

import os
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.misc import imresize
from tqdm import tqdm

#### PREPROCESS STAGE ####

# Path to superpixels class files
classes_file = "./data/ULCER_SET_CLASSES"
data_classes = pd.read_csv(classes_old, header=None)

# Instances with targets
targets = data_classes_old[1].tolist()

# Split data according to their classes
class_0 = concatenated_data[concatenated_data[1] == 0]
class_1 = concatenated_data[concatenated_data[1] == 1]
class_2 = concatenated_data[concatenated_data[1] == 2]
class_3 = concatenated_data[concatenated_data[1] == 3]

# Holdout split train/test set (Other options are k-folds or leave-one-out)
split_proportion = 0.7

split_size_0 = int(len(class_0)*split_proportion)
split_size_1 = int(len(class_1)*split_proportion)
split_size_2 = int(len(class_2)*split_proportion)
split_size_3 = int(len(class_3)*split_proportion)

new_class_0_train = np.random.choice(len(class_0), split_size_0, replace=False)
new_class_0_train = class_0.iloc[new_class_0_train]
new_class_0_test = ~class_0.iloc[:][0].isin(new_class_0_train.iloc[:][0])
new_class_0_test = class_0[new_class_0_test]

new_class_1_train = np.random.choice(len(class_1), split_size_1, replace=False)
new_class_1_train = class_1.iloc[new_class_1_train]
new_class_1_test = ~class_1.iloc[:][0].isin(new_class_1_train.iloc[:][0])
new_class_1_test = class_1[new_class_1_test]

new_class_2_train = np.random.choice(len(class_2), split_size_2, replace=False)
new_class_2_train = class_2.iloc[new_class_2_train]
new_class_2_test = ~class_2.iloc[:][0].isin(new_class_2_train.iloc[:][0])
new_class_2_test = class_2[new_class_2_test]

new_class_3_train = np.random.choice(len(class_3), split_size_3, replace=False)
new_class_3_train = class_3.iloc[new_class_3_train]
new_class_3_test = ~class_3.iloc[:][0].isin(new_class_3_train.iloc[:][0])
new_class_3_test = class_3[new_class_3_test]

x_train_list = pd.concat(
    [new_class_0_train, new_class_1_train, new_class_2_train, new_class_3_train])
x_test_list = pd.concat(
    [new_class_0_test, new_class_1_test, new_class_2_test, new_class_3_test])

# Load superpixels files
imagePath = "./data/superpixels"

x_train = []
y_train = []
for index, row in tqdm(x_train_list.iterrows(), total=x_train_list.shape[0]):
    try:
        loadedImage = plt.imread(imagePath + str(row[0]) + ".jpg")
        x_train.append(loadedImage)
        y_train.append(row[1])
    except:
        # Try with .png file format if images are not properly loaded
        try:
            loadedImage = plt.imread(imagePath + str(row[0]) + ".png")
            x_train.append(loadedImage)
            y_train.append(row[1])
        except:
            # Print file names whenever it is impossible to load image files
            print(imagePath + str(row[0]))

x_test = []
y_test = []
for index, row in tqdm(x_test_list.iterrows(), total=x_test_list.shape[0]):
    try:
        loadedImage = plt.imread(imagePath + str(row[0]) + ".jpg")
        x_test.append(loadedImage)
        y_test.append(row[1])
    except:
        # Try with .png file format if images are not properly loaded
        try:
            loadedImage = plt.imread(imagePath + str(row[0]) + ".png")
            x_test.append(loadedImage)
            y_test.append(row[1])
        except:
            # Print file names whenever it is impossible to load image files
            print(imagePath + str(row[0]))


# Reescaling of images to standard QTDU net input size and data normalization
img_width, img_height = 139, 139

index = 0
for image in tqdm(x_train):
    aux = imresize(image, (img_width, img_height, 3), "bilinear")
    x_train[index] = aux / 255.0  # Normalization
    index += 1

index = 0
for image in tqdm(x_test):
    aux = imresize(image, (img_width, img_height, 3), "bilinear")
    x_test[index] = aux / 255.0  # Normalization
    index += 1


#### TRAINING STAGE ####

os.environ["KERAS_BACKEND"] = "tensorflow"
RANDOM_STATE = 42

def get_session(gpu_fraction=0.8):

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


KTF.set_session(get_session())


def precision(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # Set F-score as 0 if there are no true positives (sklearn-like).
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


nb_classes = 4
final_model = []

# Choosing the underlying deep-model for QTDU
# Option = InceptionV3
model = InceptionV3(weights="imagenet", include_top=False,
                    input_shape=(img_width, img_height, 3))
# Option = ResNet
# model = ResNet50(weights="imagenet", include_top=False, input_shape=(3,img_width, img_height))

# Creating new outputs for the model
x = model.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(nb_classes, activation='softmax')(x)
final_model = Model(inputs=model.input, outputs=predictions)

# Metrics
learningRate = 0.001
optimizer = optimizers.SGD(lr=learningRate, momentum=0.88, nesterov=True)

# Compiling the model...
final_model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                    metrics=["accuracy", fbeta_score])

x_train = np.array(x_train)
x_test = np.array(x_test)

# Defining targets...
y_train = np.concatenate([np.full((new_class_0_train.shape[0]), 0), np.full((new_class_1_train.shape[0]), 1),
                          np.full((new_class_2_train.shape[0]), 2), np.full((new_class_3_train.shape[0]), 3)])

y_test = np.concatenate([np.full((new_class_0_test.shape[0]), 0), np.full((new_class_1_test.shape[0]), 1),
                         np.full((new_class_2_test.shape[0]), 2), np.full((new_class_3_test.shape[0]), 3)])

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

modelFilename = "./models/model_inception.h5"
trainingFilename = "./training.csv"

nb_train_samples = y_train.shape[0]
nb_test_samples = y_test.shape[0]
epochs = 10000
batch_size = 24
trainingPatience = 200
decayPatience = trainingPatience / 4

# Setting the data generator...
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    fill_mode="reflect",
    zoom_range=0.2
)

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

# Saving the model
checkpoint = ModelCheckpoint(modelFilename,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             period=1)

adaptativeLearningRate = ReduceLROnPlateau(monitor='val_acc',
                                           factor=0.5,
                                           patience=decayPatience,
                                           verbose=1,
                                           mode='auto',
                                           epsilon=0.0001,
                                           cooldown=0,
                                           min_lr=1e-8)

early = EarlyStopping(monitor='val_acc',
                      min_delta=0,
                      patience=trainingPatience,
                      verbose=1,
                      mode='auto')

csv_logger = CSVLogger(trainingFilename, separator=",", append=False)

# Callbacks
callbacks = [checkpoint, early, csv_logger, adaptativeLearningRate]

# Training of the model
final_model.fit_generator(train_generator,
                          steps_per_epoch=nb_train_samples / batch_size,
                          epochs=epochs,
                          shuffle=True,
                          validation_data=(x_test, y_test),
                          validation_steps=nb_test_samples / batch_size,
                          callbacks=callbacks)
