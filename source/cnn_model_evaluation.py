__author__ = "Baishali Dutta"
__copyright__ = "Copyright (C) 2021 Baishali Dutta"
__license__ = "Apache License 2.0"
__version__ = "0.1"

# -------------------------------------------------------------------------
#                           Importing the libraries
# -------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt_false_positive_vs_true_positive
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score, \
    roc_auc_score

# -------------------------------------------------------------------------
#                               Configurations
# -------------------------------------------------------------------------
data_dir = '../data/'
test_data_dir = data_dir + '/test/'
batch_size = 32
epochs = 100


# -------------------------------------------------------------------------
#                         Evaluating trained CNN Model
# -------------------------------------------------------------------------
def evaluate_cnn_model():
    detection_classes = ('NORMAL', 'PNEUMONIA')

    # load the trained CNN model
    cnn_model = load_model('../model/pneumonia_detection_cnn_model.h5')

    # data generator on test dataset
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        featurewise_center=True,
        featurewise_std_normalization=True)

    # preprocessing the test set
    test_dataset = test_datagen.flow_from_directory(test_data_dir,
                                                    target_size=(224, 224),
                                                    classes=detection_classes,
                                                    shuffle=False,
                                                    batch_size=batch_size)

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

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_true, y_pred_binary)
    print('Accuracy: %f' % accuracy)

    # precision tp / (tp + fp)
    precision = precision_score(y_true, y_pred_binary)
    print('Precision: %f' % precision)

    # recall: tp / (tp + fn)
    recall = recall_score(y_true, y_pred_binary)
    print('Recall: %f' % recall)

    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_true, y_pred_binary)
    print('F1 score: %f' % f1)

    # ROC AUC
    auc = roc_auc_score(y_true, y_pred_prob)
    print('ROC AUC: %f' % auc)

    # calculate roc curves
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)

    # plot the roc curve for the model
    plt.figure()
    plt_false_positive_vs_true_positive.plot(fpr, tpr, linestyle='--', label='')
    plt_false_positive_vs_true_positive.xlabel('False Positive Rate')
    plt_false_positive_vs_true_positive.ylabel('True Positive Rate')
    plt_false_positive_vs_true_positive.legend()
    plt_false_positive_vs_true_positive.show()
    plt_false_positive_vs_true_positive.savefig('ROC_Curve.jpeg')


# -------------------------------------------------------------------------
#                               Main Execution
# -------------------------------------------------------------------------
if __name__ == '__main__':
    evaluate_cnn_model()
