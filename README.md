# Multilayer Perceptron: Cancer Diagnosis With Deep Learning | 42

*This project is part of the 42 cursus AI field.*

Its purpose is to determine whether **tumors are benign** or not from a CSV file called `data.csv` that you can find in `assets` folder.

It contains information about the shape, color and texture of each tumor, and I had to use them to create **my own deep learning algorithm**, without using libs that do the work for us like *Tensorflow*.

## Neural network structure

- Input layer: **30 inputs**, but a few of them are removed before training because they are irrelevant. I evaluate it with **Kolmogorov-Smirnov test**, and remove features that create redundancy. A bias is also added to the input layer.

- Hidden layers: this multilayer perceptron contains **2 hidden layers**, each containing **16 neurons** that are activated using **Rectified Linear Unit (ReLU)**.

- Output layer: **2 neurons** are located in the output layer. The first one represents the probability of the tumor to be malignant, and the other represents the probability of a benign tumor. These neurons are activated using **softmax function**.

*NB: it would have been easier to deal with a single neuron in the output layer, but this project wanted to introce softmax function that is mostly use to classify samples into several output classes.*

## Bonuses

A few more features have been implemented to enhance the model performance, or to visualize the data:

- Training and validation loss curve plots.
- Early stopping to avoid overfitting during training.
