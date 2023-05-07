import pandas as pd
import numpy as np
import sys, os
from multilayer_perceptron import MultilayerPerceptron
sys.path.append('..')
from utils import get_df

if __name__ == '__main__':
    if not os.path.exists("../../assets/weights.npy"):
        sys.exit("Error: no weights available for prediction.")
    with open('../../assets/weights.npy', 'rb') as f:
        weights = np.load(f, allow_pickle=True)

    # Real values
    malignant_number = 212
    benign_number = 357

    df = pd.read_csv('../../assets/data_test.csv', header=None)
    df.columns = [f"Feature {i + 1}" for i in range(len(df.columns))]
    df = df.drop('Feature 13', axis=1).drop('Feature 15', axis=1)

    model = MultilayerPerceptron(df)
    model.standardize()
    model.sample = model.add_bias(model.sample)

    activations = model.get_activations(model.sample, weights)
    y_pred = [1.0 if i[0] > i[1] else 0.0 for i in activations[-1]]
    malignant_pred = y_pred.count(1.0)
    benign_pred = y_pred.count(0.0)
    print(f"Prediction results: {malignant_pred} malignant, {benign_pred} benign tumors.")
    malignant_diff = abs(malignant_pred - malignant_number) * 100 / malignant_number
    benign_diff = abs(benign_pred - benign_number) * 100 / benign_number
    print(f"Accuracy: {np.round((100 - malignant_diff + 100 - benign_diff) / 2, 2)}%") 
