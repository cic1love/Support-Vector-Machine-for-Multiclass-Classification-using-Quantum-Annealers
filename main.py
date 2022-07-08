#Copyright 2022 NTT CORPORATION

import numpy as np
import matplotlib.pyplot as plt
import math
from qsvm import OneVsRestClassifier
from qsvm import qSVM
from cimsdk.optimizers.da_optimizer import DigitalAnnealerOptimizer
from sklearn.datasets import make_classification, make_gaussian_quantiles

#hyperparameter
B = 10
K = 3
XI = 0
GAMMA = 10
C = 1
class_num = 3

def get_data_3Dsynthetic(mean, sd, num, num_dims):
    phi = np.linspace(0, 2*math.pi, num=40)
    s_nx = np.random.normal(loc=mean, scale=sd, size=num)
    s_ny = np.random.normal(loc=mean, scale=sd, size=num)

    x = np.zeros(shape=(num, num_dims))
    y = np.ones(shape=(num,))
    r_n = np.ones(shape=(num,))

    for i in range(num):
        if i < 40:
            y[i] = 0
        if 40 < i < 80:
            y[i] = 1
        if i > 80:
            y[i] = 2

    for i in range(num):
        if i < 40:
            r_n[i] = 2
        if 40 < i < 80:
            r_n[i] = 1
        if i > 80:
            r_n[i] = 0.15

    for i in range(40):
        x[i, 0] = r_n[i]*math.cos(phi[i]) + s_nx[i]
        x[i, 1] = r_n[i]*math.sin(phi[i]) + s_ny[i]

    for i in range(40):
        x[40+i, 0] = r_n[40+i]*math.cos(phi[i]) + s_nx[40+i]
        x[40+i, 1] = r_n[40+i]*math.sin(phi[i]) + s_ny[40+i]

    for i in range(40):
        x[80+i, 0] = r_n[80+i]*math.cos(phi[i]) + s_nx[80+i]
        x[80+i, 1] = r_n[80+i]*math.sin(phi[i]) + s_ny[80+i]

    np.save('data_3c.npy', x)
    np.save('label_3c.npy', y)

    return x, y

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

def main_9():

    data, label = data, label = make_classification(n_samples=60, random_state=666, n_features=2, n_redundant=0,
                                                    n_informative=2, n_clusters_per_class=1, n_classes=3)

    # graph common settings
    h = .02  # step size in the mesh
    x_min, x_max = data[:, 0].min() - .5, data[:, 0].max() + .5
    y_min, y_max = data[:, 1].min() - .5, data[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for ax_row, C in zip(axes, [0.01, 1, 100]):
        for ax, gamma in zip(ax_row, [0.1, 1, 10]):
            title = "SVM multi-class (C=%s, g=%s, Xi=%s,)" % (C, gamma, XI)
            clf = OneVsRestClassifier(class_num, qSVM, params={"data": data, "label": label, "B": B, "K": K, "Xi": XI,
                                                               "gamma": gamma, "C": C})
            scores=decision_boundary(clf, data, label, ax, title, xx, yy)
            print("SVM model Training set accuracy: ", scores, "%")

    plt.show()

    #plt.title('SVM multi-class (B=%s,K=%s,C=%s, gamma=%s, Xi=%s, acc=%f)'% (B, K, C, GAMMA, XI, scores))

def main():
    data, label = make_gaussian_quantiles(n_samples=120,n_features=2, n_classes=3, random_state=66)
    data, label = get_data_3Dsynthetic(0, 0.2, 120, 2)

    print(data, label)
    #plt.scatter(data[:, 0], data[:, 1], c=label)
    #plt.show()

    # graph common settings
    h = .02  # step size in the mesh
    x_min, x_max = data[:, 0].min() - .5, data[:, 0].max() + .5
    y_min, y_max = data[:, 1].min() - .5, data[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    C = 1
    gamma = 5

    title = "Synthetic 3 class data by SA (gamma=%s, Xi=%s,)" % (gamma, XI)
#    title = "Synthetic 3 class data(gamma=%s, Xi=%s,)" % (gamma, XI)
    clf = OneVsRestClassifier(class_num, qSVM, params={"data": data, "label": label, "B": B, "K": K, "Xi": XI,
                                                       "gamma": gamma, "C": C,
                                                       })
#                                                       "optimizer": DigitalAnnealerOptimizer})
    scores=decision_boundary(clf, data, label, axes, title, xx, yy)
    print("SVM model Training set accuracy: ", scores, "%")

    plt.show()

if __name__ == '__main__':
    main()
