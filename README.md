## Motivation

Pneumonia is a lung infection (🫁) that inflames the air sacs in one or both lungs. This infection arises when the air sacs get filled with fluid or pus (purulent material). It can be a bacterial or viral infection. The main symptoms are - cough with phlegm or pus, fever, chills, and breathing difficulty. 

This disease is responsible for over 15% of all deaths of children under five years old worldwide. This proves the severity of this disease and the need for accurate detection. 

The most commonly used method to diagnose pneumonia is through chest radiograph or chest X-ray, which depicts the infection as an increased opacity in the lungs' specific area(s).

To increase the diagnosis procedure's efficacy and reach, we can leverage machine learning algorithms to identify abnormalities in the chest X-ray images. In this model, many chest X-ray images (both normal and pneumonia) are fed to build `Convolutional Neural Network (CNN)` model for fulfilling the purpose. 


## Requirements

- Python 3.7.x
- Tensorflow 2.4.1+
- Keras 2.4.3+
- scikit-learn 0.24.1+
- matplotlib 3.3.3+
- texttable 1.6.3+
- gradio 1.5.3+

## Evaluation 

Our model is trained with 96% accuracy on the training dataset. The model's accuracy on the `test` and `val` datasets are 91% and 88% respectively. In both cases, the `f1-score` and `ROC_AUC Score` are relatively high, as shown below. 

### On Test Dataset (624 images, 234 `Normal` and 390 `Pneumonia`)

<p align="center">
<img width="960" alt="Screenshot 2021-02-07 at 17 07 23" src="https://user-images.githubusercontent.com/76659596/107152321-93f83f80-6967-11eb-95b4-0bfb3ccae6d7.png">
</p>

### On Validation Dataset (16 images, 8 `Normal` and 8 `Pneumonia`)

<p align="center">
<img width="960" alt="Screenshot 2021-02-07 at 17 10 07" src="https://user-images.githubusercontent.com/76659596/107152360-ba1ddf80-6967-11eb-90cb-dfaeca31f275.png">
</p>

## Web Application

To run the web application locally, go to the `webapp` directory and execute:

`python3 web_app.py`

This will start a local server that you can access in your browser. You can either upload/drag a new X-ray image or select any test X-ray images from the examples below.

You can, alternatively, try out the hosted web application [here](https://gradio.app/g/baishalidutta/Pneumonia-Detection).

## Developer

Chungu Chipimo Chama(<a href='mailto:chungu424@gmail.com'>chungu424@gmail.com</a>)


