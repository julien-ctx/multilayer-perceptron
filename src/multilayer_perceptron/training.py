import pandas as pd
import numpy as np
import sys, os
from multilayer_perceptron import MultilayerPerceptron
sys.path.append('..')
from utils import get_df, color

if __name__ == '__main__':
	if len(sys.argv) != 2:
		sys.exit(f"{color.RED}Error: invalid number of arguments.{color.END}")

	df = get_df(sys.argv[1])
 
	model = MultilayerPerceptron(df)

	# Features engineering
	model.drop_irrelevant_data()
	model.standardize()
	
	model.fit()
