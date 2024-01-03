import numpy as np
import pandas as pd
from scipy.io.arff import loadarff 

raw_data = loadarff('dataset.arff')
df_data = pd.DataFrame(raw_data[0])

# print(df_data)

# write df_data into .out file
# turn into integer before writing
# blank space between each number

#drop the first column

df_data = df_data.drop(df_data.columns[0], axis=1)

df_data = df_data.astype(float)

# sort with respect to the last column

df_data = df_data.sort_values(by=['Class'])

# store the last column in a separate variable

df_class = df_data['Class']

# remove the last column (80 features and 1 class in a row)
# drop the column "Class" from df_data

df_data = df_data.drop(df_data.columns[80], axis=1)

# drop and store 3 columns randomly
# shuffle and drop the first column
df_data = df_data.sample(frac=1).reset_index(drop=True)
# store the first column in a separate variable

df_data1 = df_data[df_data.columns[0]]

# shuffle the columns

df_data = df_data.drop(df_data.columns[0], axis=1)
df_data = df_data.sample(frac=1).reset_index(drop=True)

# store the second column in a separate variable

df_data2 = df_data[df_data.columns[0]]

df_data = df_data.drop(df_data.columns[0], axis=1)
df_data = df_data.sample(frac=1).reset_index(drop=True)

# store the third column in a separate variable

df_data3 = df_data[df_data.columns[0]]

# add the last column back

df_data = pd.concat([df_data1, df_data2, df_data3], axis=1)

df_data['Class'] = df_class


# remove the first column


df_data = df_data.sample(frac=1).reset_index(drop=True)

df_data.to_csv('data.in', sep=' ', index=False, header=False)

# remove the last column

df_data = df_data.drop(df_data.columns[3], axis=1)

# use svm to classify the data

from sklearn import svm

# split the data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_data, df_class, test_size=0.2, random_state=0)

# need a copy of the training set

X_train_copy = X_train.copy()

# train the model

# use non-linear soft margin classifier

clf = svm.SVC(kernel='rbf', C=1.0, gamma=0.1)

clf.fit(X_train, y_train)

# predict the training set

y_train_pred = clf.predict(X_train)

#y_pred = clf.predict(X_test)

# predict the test set

print(y_train_pred)

# calculate the accuracy

from sklearn.metrics import accuracy_score

print(accuracy_score(y_train, y_train_pred))