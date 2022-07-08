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

def decision_boundary(clf, X, y, ax, title,xx,yy):
    clf.solve(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)

    # label
    ax.set_title(title)

    scores = clf.evaluate(X, y)

    return scores

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