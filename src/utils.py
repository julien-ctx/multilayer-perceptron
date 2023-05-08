import pandas as pd
import matplotlib.pyplot as plt
import os, sys

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96;1m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94;1m'
   GREEN = '\033[92;1m'
   YELLOW = '\033[93;1m'
   RED = '\033[91;1m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def get_df(df_name):
	if not os.path.exists(df_name):
		sys.exit(f"{color.RED}Error: dataset doesn't exist{color.END}")
	df = pd.read_csv(df_name, header=None)
	df = df.drop(df.columns[0], axis=1)
	df.columns = [f"Feature {i}" if i > 0 else "Diagnosis" for i in range(len(df.columns))]
	df['Diagnosis'] = df['Diagnosis'].replace('M', 1.0).replace('B', 0.0)
	return df
