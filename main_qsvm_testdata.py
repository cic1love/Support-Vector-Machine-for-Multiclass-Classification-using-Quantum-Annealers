#Copyright 2022 NTT CORPORATION

import numpy as np
import matplotlib.pyplot as plt
from qsvm import OneVsRestClassifier
from qsvm import qSVM
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv("Sets_Data1.csv").to_numpy()

X = data[:, 0:26]
y = data[:, 27]

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=2)

#hyperparameter
B = 10
K = 3
XI = 0
GAMMA = 10
C = 1
class_num = 4

def main():
    data, label = train_x, train_y

    print(data, label)

    clf = OneVsRestClassifier(class_num, qSVM, params={"data": data, "label": label, "B": B, "K": K, "Xi": XI,
                                                       "gamma": GAMMA, "C": C})
    clf.solve(data, label)

    scores_train = clf.evaluate(data, label)
    scores_test = clf.evaluate(test_x, test_y)

    print("scores_train", scores_train)
    print("scores_test", scores_test)

if __name__ == '__main__':
    main()
