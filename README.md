<p align="center">
  <img width="621" alt="logo" src="https://user-images.githubusercontent.com/76659596/107152793-30234600-696a-11eb-8827-56cb0c3a7578.png">
</p>

[![online hosted application](https://img.shields.io/static/v1?label=Try-It-Out&message=Online-Hosted-Application&color=yellow&logo=godot-engine)](https://gradio.app/g/baishalidutta/Pneumonia-Detection)
[![License - Apache-2.0](https://img.shields.io/badge/License-Apache-blue)](#license)
[![Build - Passing](https://img.shields.io/badge/Build-Passing-brightgreen)](https://github.com/baishalidutta/Pneumonia-Detection/runs/1485969918)
[![GitHub release](https://img.shields.io/github/release/baishalidutta/Pneumonia-Detection?include_prereleases&sort=semver)](https://github.com/baishalidutta/Pneumonia-Detection/releases/)
![Python](https://img.shields.io/badge/python-v3.7.x-blue.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/baishalidutta/Pneumonia-Detection/issues)

<p align="center">
  <img width="621" alt="web-app-screencast" src="https://user-images.githubusercontent.com/76659596/107169811-14965a80-69bf-11eb-8f04-df3965eea358.gif">
</p>

## Motivation

Pneumonia is a lung infection (🫁) that inflames the air sacs in one or both lungs. This infection arises when the air sacs get filled with fluid or pus (purulent material). It can be a bacterial or viral infection. The main symptoms are - cough with phlegm or pus, fever, chills, and breathing difficulty. 

This disease is responsible for over 15% of all deaths of children under five years old worldwide. This proves the severity of this disease and the need for accurate detection. 

The most commonly used method to diagnose pneumonia is through chest radiograph or chest X-ray, which depicts the infection as an increased opacity in the lungs' specific area(s).

To increase the diagnosis procedure's efficacy and reach, we can leverage machine learning algorithms to identify abnormalities in the chest X-ray images. In this model, many chest X-ray images (both normal and pneumonia) are fed to build `Convolutional Neural Network (CNN)` model for fulfilling the purpose. 

## Requirements

- Python 3.7.0+
- Tensorflow 2.4.1+
- Keras 2.4.3+
- scikit-learn 0.24.1+
- matplotlib 3.3.3+
- texttable 1.6.3+
- gradio 1.5.3+

## Dataset

You can download the dataset from [kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/). Use the underlying download link to download the dataset.

### Instructions to follow

* Extract the archive
* You will find several directories in it
* Copy the `chest-xray` directory contents (`train`, `test` and `val` subdirectories) to the `data` folder

The number of images belonging to both classes (`Normal` and `Pneumonia`) in the `train`, `test` and `val` datasets are -

| Dataset Type | Normal | Pneumonia |
|--------------|--------|-----------|
| Training     | 1341   | 3875      |
| Test         | 234    | 390       |
| Validation   | 8      | 8         |


## Installation

* Clone the repository 

`git clone https://github.com/baishalidutta/Pneumonia-Detection.git`

* Install the required libraries

`pip3 install -r requirements.txt`

## Usage

Enter into the `source` directory to execute the following source codes.

* To generate the model on your own, run

`python3 model_training.py` 

* To evaluate any dataset using the pre-trained model (in the `model` directory), run

`python3 model_evaluation.py`

Note that, for evaluation, `model_evaluation.py` will use all the images contained inside both `test` and `val` subdirectories (inside `data` directory).

Alternatively, you can find the whole analysis in the notebook inside the `notebook` directory. To open the notebook, use either `jupyter notebook` or `google colab` or any other IDE that supports notebook feature such as `PyCharm Professional`.

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

Baishali Dutta (<a href='mailto:me@itsbaishali.com'>me@itsbaishali.com</a>)

## Contribution [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/baishalidutta/Pneumonia-Detection/issues)

If you would like to contribute and improve the model further, check out the [Contribution Guide](https://github.com/baishalidutta/Pneumonia-Detection/blob/main/CONTRIBUTING.md)

## License [![License](http://img.shields.io/badge/license-Apache-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

This project is licensed under Apache License Version 2.0
