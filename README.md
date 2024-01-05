# Handwritten English Word Recognition

## Introduction

This project focuses on Handwritten English Word Recognition using deep learning techniques. The goal is to develop a model capable of accurately recognizing handwritten words. The project utilizes a dataset of handwritten English words for training, validation, and testing.

## Dataset

We have used <a href="https://fki.tic.heia-fr.ch/databases/iam-handwriting-database" target="_blank">IAM Handwriting Database</a> for out project.  
<p align="center"><img width="400" height="100%" src="https://github.com/PritomPaul99/Handwritten-English-Word-Recognition-Using-CNN-RNN-Model/blob/main/images/IAM_dataset_structure.png?raw=true" alt="IAM Dataset tree Structure"/></p>  
The portion of the dataset we have used in this project consists of handwritten English words. It has been organized and prepared for training, validation, and testing. The dataset is sourced from the IAM Words dataset, and details on data collection and processing can be found in the provided Jupyter notebook.

## Data Preprocessing

The dataset is split into training, validation, and test subsets. The images are preprocessed to ensure uniformity, including distortion-free resizing and character label vectorization. The character vocabulary is established using StringLookup layers from TensorFlow.

## Model Architecture

The model architecture is designed for Handwritten English Word Recognition. It includes convolutional and recurrent neural network layers. The Connectionist Temporal Classification (CTC) loss is utilized as an endpoint layer. The model is trained with an Adam optimizer.

## Evaluation Metric

The project employs the Edit Distance as the evaluation metric. An Edit Distance Callback is implemented to monitor the model's performance during training.

## Training

The model is trained over a specified number of epochs, with progress and evaluation metrics monitored. The Edit Distance Callback provides insights into the model's ability to recognize handwritten words. After training, the model's performance is evaluated on the test dataset.

## Inference

Inference is demonstrated using the trained model on a sample from the test dataset. Predictions are generated, and the corresponding handwritten words are visualized alongside the original images.

## Usage

- Install Dependencies: Ensure that the required dependencies, as mentioned in the Jupyter notebook, are installed.
- Data Collection: The dataset is downloaded and organized using the provided commands.
- Data Preprocessing: Execute the code cells for data preprocessing to prepare the dataset.
- Model Training: Train the model by executing the code cells for model building and training.
- Inference: Use the trained model for inference on new handwritten word images.

## Results

The project aims to achieve accurate recognition of handwritten English words. The model's performance is evaluated using the Edit Distance metric, providing insights into the accuracy of predictions.

<p align="center"><img width="100%" height="100%" src="https://github.com/PritomPaul99/Handwritten-English-Word-Recognition-Using-CNN-RNN-Model/blob/main/images/output1.png?raw=true" alt="Result/Output"/></p>

## Conclusion

Handwritten English Word Recognition is a challenging task, and this project addresses it using deep learning techniques. The provided documentation guides users through dataset preparation, model training, and inference, enabling them to understand, replicate, and extend the project.
