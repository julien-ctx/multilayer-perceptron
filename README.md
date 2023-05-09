# Multilayer Perceptron: Cancer Diagnosis With Deep Learning | 42

*This project is part of the 42 cursus AI field.*

Its purpose is to determine whether **tumors are benign** or not from a CSV file called `data.csv` that you can find in `assets` folder.

It contains information about the shape, color and texture of each tumor, and I had to use them to create **my own deep learning algorithm**, without using libs that do the work for us like *Tensorflow*.

## Usage

Install the required libraries with `pip3 install -r requirements.txt`

Go to `src/multilayer_perceptron`. In this folder, you can train the model and make predictions.
- `python3 training.py ../../assets/data.csv` to train with `data.csv` (you can change the features file).
- `python3 prediction.py ../../assets/data.csv` to predict on `data.csv` with saved weights and used features from the training phase.

## Neural network structure

- Input layer: **30 inputs**, but a few of them are removed before training because they are irrelevant. I evaluate it with **Kolmogorov-Smirnov test**, and remove features that create redundancy. A bias is also added to the input layer.

- Hidden layers: this multilayer perceptron contains **2 hidden layers**, each containing **16 neurons** that are activated using **Rectified Linear Unit (ReLU)**.

- Output layer: **2 neurons** are located in the output layer. The first one represents the probability of the tumor to be malignant, and the other represents the probability of a benign tumor. These neurons are activated using **softmax function**.

*NB: it would have been easier to deal with a single neuron in the output layer, but this project wanted to introduce softmax function that is mostly use to classify samples into several output classes.*

## Training and loss optimization

This deep learning model relies on a **gradient descent algorithm**. We look to minimize the error using the **binary cross-entropy loss function** derivative.

`data.csv` has been split into two parts:
- 80% for training.
- 20% for validation.

In this project, I realized that the key to avoid overfitting is to use **Early Stopping** and stop iterations whenever the validation loss stagnates or starts to decrease.

Hyperparameters are also important, and I ended up using `α = 0.001` and `epochs = 200`. This is how I managed to get the best from my model.

## Results

`.npy` files are used to store weights and irrelevant features so that they can be removed on test datasets too.

For data.csv, **Accuracy Score** is 100% and **Loss** is 0.0598.

On unknow datasets, the model performs pretty well with Accuracy Score above 90% and Loss less than 0.1.

## Bonuses

A few more features have been implemented to enhance the model performance, or to visualize the data:

- Data visualization with histograms, box plots and an heatmap.
- Early stopping to avoid overfitting during training.
- Stochastic Gradient Descent.
