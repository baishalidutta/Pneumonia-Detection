__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Importing the libraries
# -------------------------------------------------------------------------
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, \
    classification_report, precision_score, recall_score, f1_score, roc_auc_score

# -------------------------------------------------------------------------
#                               Configurations
# -------------------------------------------------------------------------
MODEL_LOC = '../model/pneumonia_detection_cnn_model.h5'
DATA_DIR = '../data/'
TEST_DATA_DIR = DATA_DIR + '/test/'
BATCH_SIZE = 32
EPOCHS = 100
DETECTION_CLASSES = ('NORMAL', 'PNEUMONIA')


# -------------------------------------------------------------------------
#                         Evaluating trained CNN Model
# -------------------------------------------------------------------------
def evaluate_cnn_model():
    """
    Loads the pre-trained model and executes the model on the test data.
    Prints the model evaluation results and plots the ROC curve
    """
    # load the trained CNN model
    cnn_model = load_model(MODEL_LOC)

    # data generator on test dataset
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        featurewise_center=True,
        featurewise_std_normalization=True)

    # preprocessing the test set
    test_dataset = test_datagen.flow_from_directory(TEST_DATA_DIR,
                                                    target_size=(224, 224),
                                                    classes=DETECTION_CLASSES,
                                                    shuffle=False,
                                                    batch_size=BATCH_SIZE)

    # storing the true classes of the test dataset
    y_true = test_dataset.classes

    # predicting the classes of the test dataset
    y_pred = cnn_model.predict_generator(test_dataset, steps=len(test_dataset), verbose=1)
    # Storing the predicted probability
    y_pred_prob = y_pred[:, 1]
    # Storing the binary classes for the predictions
    y_pred_binary = y_pred_prob > 0.5

    # -------------------------------------------------------------------------
    #                         Model Evaluation Matrices
    # -------------------------------------------------------------------------
    # confusion Matrix
    print('\nConfusion Matrix\n -------------------------')
    print(confusion_matrix(y_true, y_pred_binary))

    # classification report
    # accuracy: (tp + tn) / (p + n)
    # precision tp / (tp + fp)
    # recall: tp / (tp + fn)
    # f1_score: 2 tp / (2 tp + fp + fn)
    print('\nClassification Report\n -------------------------')
    print(classification_report(y_true, y_pred_binary))

    # ROC AUC
    auc = roc_auc_score(y_true, y_pred_prob)
    print('ROC AUC: %f' % auc)

    # calculate roc curves
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)

    # plot the roc curve for the model
    plt.figure()
    plt.plot(fpr, tpr, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    plt.savefig("../plots/ROC_Curve.jpeg")


# -------------------------------------------------------------------------
#                               Main Execution
# -------------------------------------------------------------------------
if __name__ == '__main__':
    evaluate_cnn_model()
