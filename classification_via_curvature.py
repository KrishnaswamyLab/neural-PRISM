from os import listdir
import numpy as np
import phate
import scipy
from tqdm import tqdm


import time
from numpy import linalg as LA
from tqdm import tqdm
import matplotlib.pyplot as plt
import tphate
import scprep

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#hold out one patient

band = 'alpha'

X1 = np.load(f'data/Curvature/{band}_G{1}_cond_stack_all_cur.npy')
X2 = np.load(f'data/Curvature/{band}_G{2}_cond_stack_all_cur.npy')

cut_off = int(0.05 * X1.shape[1])
full = X1.shape[1]
select_num = int((X1.shape[1] - 2 * cut_off) * 0.1)
X1 = X1[:, cut_off: full-cut_off][:, :: 30]
X2 = X2[:, cut_off: full-cut_off][:, :: 30]

Y1 = np.zeros((X1.shape[0],))
Y2 = np.ones((X2.shape[0],))
y = np.concatenate((Y1, Y2), axis=0)


sublist1 = [1,2,3, 4,5, 6, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21]
sublist2 = [1,2,4,5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]

non_exist1 = [5, 6]
non_exist2 = [12, 15]

len_non_exist1 = len(non_exist1) * 24 - 7


acc = []
for i, sub_1 in tqdm(enumerate(sublist1)):
    if sub_1 not in non_exist1:
        for k in range(8):
            cut_start1 = (i-6) * 24 + len_non_exist1
            cut_end1 = (i-6) * 24 + len_non_exist1 + 24
            cut_start2 = 24 * k
            cut_end2 = 24 * (k+1)
            X_train = np.concatenate((X1[: cut_start1], X1[cut_end1:], X2[:cut_start2], X2[cut_end2:]), axis = 0)
            X_test = np.concatenate((X1[cut_start1:cut_end1], X2[cut_start2:cut_end2]), axis = 0)
            y_train = np.concatenate((Y1[: cut_start1], Y1[cut_end1:], Y2[:cut_start2], Y2[cut_end2:]), axis = 0)
            y_test = np.concatenate((Y1[cut_start1:cut_end1], Y2[cut_start2:cut_end2]), axis = 0)
        
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            mlp = MLPClassifier(hidden_layer_sizes=(32, 64, 32), max_iter=5000, random_state=42)
            mlp.fit(X_train, y_train)
 
            # Make predictions on the test data
            y_pred = mlp.predict(X_test)
 
            # Calculate the accuracy of the model
            accuracy = accuracy_score(y_test, y_pred)
            acc.append(accuracy)
acc = np.array(acc)


print(np.mean(acc))
