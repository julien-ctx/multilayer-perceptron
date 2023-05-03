import pandas as pd
import numpy as np
import sys, os
from multilayer_perceptron import MultilayerPerceptron
sys.path.append('..')
from utils import get_df

if __name__ == '__main__':
    if not os.path.exists("weights.csv"):
        sys.exit("Error: no weights available for prediction.")
