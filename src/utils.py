import pandas as pd
import matplotlib.pyplot as plt
import os, sys

def get_df():
	if not os.path.exists('../../assets/data.csv'):
		sys.exit("Error: data.csv doesn't exist")
	df = pd.read_csv('../../assets/data.csv')
	df.columns = [f"Feature {i + 1}" if i < 1 else f"Feature {i}" if i > 1 else "Malignant" for i in range(len(df.columns))]
	df['Malignant'] = df['Malignant'].replace('M', 1.0).replace('B', 0.0)
	# Add malignant at the beginning of df.
	tmp = df['Malignant']
	df.pop('Malignant')
	df.insert(0, 'Malignant', tmp)
	return df
