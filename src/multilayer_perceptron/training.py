import pandas as pd
import numpy as np
import sys, os
from multilayer_perceptron import MultilayerPerceptron
sys.path.append('..')
from utils import get_df

if __name__ == '__main__':
	if os.path.exists("weights.csv"):
		os.remove("weights.csv")
	df = get_df()
 
	model = MultilayerPerceptron(df)

	# Features engineering
	model.drop_irrelevant_data()
	model.standardize()
	
	model.fit()
