import pandas as pd
import numpy as np
import sys, os
from multilayer_perceptron import MultilayerPerceptron
sys.path.append('..')
from utils import *

if __name__ == '__main__':
	if len(sys.argv) == 1 or len(sys.argv) > 3:
		sys.exit(f"{color.RED}Error: invalid number of arguments.{color.END}")

	df = get_df(sys.argv[1])
 
	if len(sys.argv) == 2:
		model = MultilayerPerceptron(df, algo.GD.value)
	else:
		model = MultilayerPerceptron(df, sys.argv[2])

	# Features engineering
	model.drop_irrelevant_data()
	model.standardize()
	
	model.fit()
