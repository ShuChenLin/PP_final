import numpy as np
import pandas as pd
from scipy.io.arff import loadarff 

raw_data = loadarff('phpjX67St.arff')
df_data = pd.DataFrame(raw_data[0])

print(df_data)

# write df_data into .out file
# turn into integer before writing
# blank space between each number

df_data = df_data.astype(int)

# sort with respect to the last column

df_data = df_data.sort_values(by=['Class'])

# only keep the first 200 rows
# shuffle the rows

df_data = df_data.iloc[:200]
df_data = df_data.sample(frac=1).reset_index(drop=True)

df_data.to_csv('data.in', sep=' ', index=False, header=False)