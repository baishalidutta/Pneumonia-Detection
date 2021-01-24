__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Importing the libraries
# -------------------------------------------------------------------------
import os

import matplotlib.pyplot as plt
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

# MacOS specific issue for OpenMP runtime (workaround)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# -------------------------------------------------------------------------
#                               Configurations
# -------------------------------------------------------------------------
MODEL_LOC = '../model/pneumonia_detection_cnn_model.h5'
DATA_DIR = '../data/'
TRAINING_DATA_DIR = DATA_DIR + '/train/'
VALIDATION_DATA_DIR = DATA_DIR + '/val/'
DETECTION_CLASSES = ('NORMAL', 'PNEUMONIA')
BATCH_SIZE = 32
EPOCHS = 100


# -------------------------------------------------------------------------
#                   Building the CNN Model Architecture
# -------------------------------------------------------------------------
def build_cnn_model():
    """
    Specifies the CNN architecture which consists of the following steps
    :return: the CNN model
    """
    cnn_model = Sequential()

    # First Block of CNN
    cnn_model.add(Conv2D(32, (3, 3), padding='same', input_shape=(224, 224, 3)))
    cnn_model.add(Activation('relu'))
    cnn_model.add(MaxPooling2D((2, 2)))

    #  Second Block of CNN
    cnn_model.add(Conv2D(64, (3, 3), padding='same'))
    cnn_model.add(Activation('relu'))
    cnn_model.add(MaxPooling2D((2, 2)))

    #  Third Block of CNN
    cnn_model.add(Conv2D(128, (3, 3), padding='same'))
    cnn_model.add(Activation('relu'))
    cnn_model.add(MaxPooling2D((2, 2)))

    #  Flatten and Fully Connected Layer
    cnn_model.add(Flatten())
    cnn_model.add(Dense(1000))
    cnn_model.add(Activation('relu'))

    #  Softmax Classifier
    cnn_model.add(Dense(2))
    cnn_model.add(Activation('softmax'))

    #  Display model
    cnn_model.summary()

    # compile model
    opt = SGD(learning_rate=0.001)
    cnn_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return cnn_model


# -------------------------------------------------------------------------
#                Data Preprocessing and CNN Model Training
# -------------------------------------------------------------------------
def train_cnn_model(cnn_model):
    """
    Trains the CNN model on the training and validation dataset
    :param cnn_model: the CNN model
    :return: the training history
    """
    # data generator on training dataset
    training_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        featurewise_center=True,
        featurewise_std_normalization=True)

    # preprocessing the training set
    training_dataset = training_datagen.flow_from_directory(TRAINING_DATA_DIR,
                                                            classes=DETECTION_CLASSES,
                                                            batch_size=BATCH_SIZE,
                                                            target_size=(224, 224))

    # data generator on validation dataset
    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        featurewise_center=True,
        featurewise_std_normalization=True)

    # preprocessing the validation set
    validation_dataset = validation_datagen.flow_from_directory(VALIDATION_DATA_DIR,
                                                                classes=DETECTION_CLASSES,
                                                                batch_size=BATCH_SIZE,
                                                                target_size=(224, 224))

    # early stopping
    early_stop = EarlyStopping(monitor='val_accuracy',
                       mode='max',
                       patience=10)

    history = cnn_model.fit_generator(training_dataset,
                                      steps_per_epoch=len(training_dataset),
                                      validation_data=validation_dataset,
                                      validation_steps=len(validation_dataset),
                                      epochs=EPOCHS,
                                      callbacks=[early_stop],
                                      verbose=1)
    # save the CNN model
    cnn_model.save(MODEL_LOC)

    return history


# -------------------------------------------------------------------------
#    Plotting the training history for further hyperparameter tuning
# -------------------------------------------------------------------------
def plot_training_history(history):
    """
    Generates plots for accuracy and loss
    :param history: the training history
    :return: generated plots
    """
    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig("../plots/accuracy.jpeg")

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig("../plots/loss.jpeg")


# -------------------------------------------------------------------------
#                               Main Execution
# -------------------------------------------------------------------------
def execute():
    cnn_model = build_cnn_model()
    history = train_cnn_model(cnn_model)
    plot_training_history(history)


if __name__ == '__main__':
    execute()
