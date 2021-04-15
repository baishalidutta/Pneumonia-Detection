__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import the libraries
# -------------------------------------------------------------------------
import os

import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D, \
    Dense, Flatten, BatchNormalization, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from config import *

# MacOS specific issue for OpenMP runtime (workaround)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# -------------------------------------------------------------------------
#                      Build CNN Model Architecture
# -------------------------------------------------------------------------
def build_cnn_model():
    """
    Specifies the CNN architecture which consists of the following steps
    :return: the CNN model
    """
    cnn_model = Sequential()

    # First Block of CNN
    cnn_model.add(Conv2D(16, (3, 3), padding='same', input_shape=(224, 224, 3), activation='relu'))
    cnn_model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))

    #  Second Block of CNN
    cnn_model.add(SeparableConv2D(32, (3, 3), padding='same', activation='relu'))
    cnn_model.add(SeparableConv2D(32, (3, 3), padding='same', activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((2, 2)))

    #  Third Block of CNN
    cnn_model.add(SeparableConv2D(64, (3, 3), padding='same', activation='relu'))
    cnn_model.add(SeparableConv2D(64, (3, 3), padding='same', activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((2, 2)))

    #  Fourth Block of CNN
    cnn_model.add(SeparableConv2D(128, (3, 3), padding='same', activation='relu'))
    cnn_model.add(SeparableConv2D(128, (3, 3), padding='same', activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Dropout(rate=0.2))

    #  Fifth Block of CNN
    cnn_model.add(SeparableConv2D(256, (3, 3), padding='same', activation='relu'))
    cnn_model.add(SeparableConv2D(256, (3, 3), padding='same', activation='relu'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Dropout(rate=0.2))

    #  Flatten and Fully Connected Layer
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units=512, activation='relu'))
    cnn_model.add(Dropout(rate=0.7))
    cnn_model.add(Dense(units=128, activation='relu'))
    cnn_model.add(Dropout(rate=0.5))
    cnn_model.add(Dense(units=64, activation='relu'))
    cnn_model.add(Dropout(rate=0.3))

    #  Softmax Classifier
    cnn_model.add(Dense(units=2, activation='softmax'))

    #  Display model
    cnn_model.summary()

    # Compile model
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
    # data generator on training dataset, data augmentation applied
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        shear_range=0.2,
        vertical_flip=True,
        rotation_range=10,
        zoom_range=0.3)

    # preprocessing the training set
    training_dataset = train_datagen.flow_from_directory(TRAINING_DATA_DIR,
                                                         classes=DETECTION_CLASSES,
                                                         shuffle=True,
                                                         batch_size=BATCH_SIZE,
                                                         target_size=(224, 224))

    # data generator on test dataset (here used as validation)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # preprocessing the test set (here used as validation)
    test_dataset = test_datagen.flow_from_directory(TEST_DATA_DIR,
                                                    classes=DETECTION_CLASSES,
                                                    shuffle=False,
                                                    batch_size=BATCH_SIZE,
                                                    target_size=(224, 224))

    # introducing callbacks
    early_stop = EarlyStopping(monitor='val_loss',
                               patience=10,
                               mode='min',
                               min_delta=0.001,
                               restore_best_weights=True)

    checkpoint = ModelCheckpoint(filepath=MODEL_LOC,  # saves the 'best' model
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='min')

    # fit the model
    history = cnn_model.fit(training_dataset,
                            steps_per_epoch=len(training_dataset),
                            validation_data=test_dataset,
                            validation_steps=len(test_dataset),
                            epochs=EPOCHS,
                            callbacks=[early_stop, checkpoint],
                            verbose=1)

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
    plt.savefig("../plots/accuracy.jpeg")
    plt.show()

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("../plots/loss.jpeg")
    plt.show()


def execute():
    cnn_model = build_cnn_model()
    history = train_cnn_model(cnn_model)
    plot_training_history(history)


# -------------------------------------------------------------------------
#                               Main Execution
# -------------------------------------------------------------------------
if __name__ == '__main__':
    execute()
