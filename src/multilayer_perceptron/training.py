import pandas as pd
import numpy as np
import sys
from multilayer_perceptron import MultilayerPerceptron
sys.path.append('..')
from utils import get_df

if __name__ == '__main__':
	df = get_df()
 
	model = MultilayerPerceptron(df)

	# Features engineering
	model.drop_irrelevant_data()
	model.standardize()
