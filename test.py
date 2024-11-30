import pandas as pd

file_path = 'X_train_Hi5.csv'

column_names = pd.read_csv(file_path, nrows=0).columns.tolist()
print(f"Column at position 1: {column_names[1]}")
print(f"Column at position 5: {column_names[5]}")