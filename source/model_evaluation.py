__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Import Libraries
# -------------------------------------------------------------------------
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, confusion_matrix, \
    ConfusionMatrixDisplay, classification_report, roc_auc_score

from config import *


# -------------------------------------------------------------------------
#                         Evaluate Trained CNN Model
# -------------------------------------------------------------------------
def evaluate_cnn_model(evaluation_directory, dataset_type):
    """
    Loads the pre-trained model and executes the model on the test data.
    Prints the model evaluation results and plots the ROC curve
    """
    # load the trained CNN model
    cnn_model = load_model(MODEL_LOC)

    # data generator on test dataset (no data augmentation applied)
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # preprocess the test set
    dataset = datagen.flow_from_directory(evaluation_directory,
                                          target_size=(224, 224),
                                          classes=DETECTION_CLASSES,
                                          shuffle=False,
                                          batch_size=BATCH_SIZE)

    # store the true classes of the test dataset
    y_true = dataset.classes

    # predict the classes of the test dataset
    y_pred = cnn_model.predict(dataset, steps=len(dataset), verbose=1)

    # store the predicted probability
    y_pred_prob = y_pred[:, 1]

    # store the binary classes for the predictions
    y_pred_binary = y_pred_prob > 0.5

    # -------------------------------------------------------------------------
    #                         Model Evaluation Matrices
    # -------------------------------------------------------------------------
    # print and display the confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=DETECTION_CLASSES)
    cm_display.plot(cmap='Blues', colorbar=False)

    print(f'------------- Confusion Matrix for {dataset_type} -------------')
    print(cm)

    plt.title('Confusion Matrix')
    plt.savefig(f'../plots/confusion_matrix_{dataset_type}.jpeg')

    # display the confusion matrix

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
    plt.savefig(f'../plots/ROC_Curve_{dataset_type}.jpeg')
    plt.show()


# -------------------------------------------------------------------------
#                               Main Execution
# -------------------------------------------------------------------------
if __name__ == '__main__':
    evaluate_cnn_model(TEST_DATA_DIR, 'Test_Dataset')
    evaluate_cnn_model(VAL_DATA_DIR, 'Validation_Dataset')
