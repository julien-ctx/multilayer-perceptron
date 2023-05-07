import pandas as pd
import matplotlib.pyplot as plt
import os, sys

def get_df():
	if not os.path.exists('../../assets/data.csv'):
		sys.exit("Error: data.csv doesn't exist")
	df = pd.read_csv('../../assets/data.csv', header=None)
	df = df.drop(df.columns[0], axis=1)
	df.columns = [f"Feature {i}" if i > 0 else "Diagnosis" for i in range(len(df.columns))]
	df['Diagnosis'] = df['Diagnosis'].replace('M', 1.0).replace('B', 0.0)
	return df
