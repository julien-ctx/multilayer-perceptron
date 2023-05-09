import pandas as pd
import numpy as np
import sys, os
from multilayer_perceptron import MultilayerPerceptron
sys.path.append('..')
from utils import *

def get_real_diagnosis(df):
    y_true = df['Diagnosis']
    return (y_true == 1.0).sum(), (y_true == 0.0).sum(), y_true

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(f"{color.RED}Error: invalid number of arguments.{color.END}")
    if not os.path.exists("../../assets/weights.npy") or not os.path.exists("../../assets/features.npy"):
        sys.exit(f"{color.RED}Error: no weights or features to drop available for prediction.{color.END}")
    with open('../../assets/weights.npy', 'rb') as f:
        weights = np.load(f, allow_pickle=True)
    with open('../../assets/features.npy', 'rb') as f:
        features_to_drop = np.load(f, allow_pickle=True)

    df = get_df(sys.argv[1])

    # Real values
    malignant_number, benign_number, y_true = get_real_diagnosis(df)
    model = MultilayerPerceptron(df, algo.GD.value)
    for feature in features_to_drop:
        model.sample = model.sample.drop(feature, axis=1)
    diagnosis = model.sample['Diagnosis']
    diagnosis = np.array([[1.0, 0.0] if diag == 1 else [0.0, 1.0] for diag in diagnosis])
    model.sample = model.sample.drop('Diagnosis', axis=1)
    model.standardize()
    model.sample = model.add_bias(model.sample)

    activations = model.get_activations(model.sample, weights)
    y_pred = [1.0 if i[0] > i[1] else 0.0 for i in activations[-1]]

    malignant_pred = y_pred.count(1.0)
    benign_pred = y_pred.count(0.0)

    print(f"{color.BOLD}Prediction results: {malignant_pred} malignant, {benign_pred} benign tumors.{color.END}")
    print(f"{color.BOLD}Real results: {malignant_number} malignant, {benign_number} benign tumors.{color.END}")

    malignant_diff = abs(malignant_pred - malignant_number) * 100 / malignant_number
    benign_diff = abs(benign_pred - benign_number) * 100 / benign_number

    print(f"{color.BOLD}Accuracy: {color.GREEN}{np.round((100 - malignant_diff + 100 - benign_diff) / 2, 2)}%{color.END}") 
    print(f"{color.BOLD}Loss: {color.GREEN}{model.loss(activations[-1], diagnosis)}{color.END}") 
