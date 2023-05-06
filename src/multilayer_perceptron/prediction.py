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
    df = pd.read_csv('../../assets/data_test.csv', )
    df.columns = [f"Feature {i + 1}" for i in range(len(df.columns))]
    df = df.drop('Feature 13', axis=1).drop('Feature 15', axis=1)
    model = MultilayerPerceptron(df)
    model.standardize()
    model.sample = model.add_bias(model.sample)
    activations = model.get_activations(model.sample, weights)
    m = ['M' if i[0] > i[1] else 'B' for i in activations[-1]]
    print(m.count('M'))
    print(m.count('B'))
    # print(activations[-1])
