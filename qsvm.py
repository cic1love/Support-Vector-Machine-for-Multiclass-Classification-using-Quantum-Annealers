#Copyright 2022 NTT CORPORATION

import numpy as np
from pyqubo import Array, Binary , Placeholder,Constraint, Sum, solve_ising, solve_qubo
from cimsdk.metasolvers.qubo_metasolver import QUBOMetasolver
from itertools import combinations
from cimsdk.optimizers.sa_optimizer import SimulatedAnnealingOptimizer
from cimsdk.optimizers.da_optimizer import DigitalAnnealerOptimizer
from sklearn.metrics import accuracy_score
import math
import operator
import cimsdk


class OneVsRestClassifier:
    def __init__(self, class_num, classifier, params=None):
        """binary classifierを受け取り、クラスの数分インスタンスを作成する."""
        self.class_num = class_num
        if params is None:
            self.classifiers = [classifier() for _ in range(class_num)]
        else:
            self.classifiers = [classifier(**params) for _ in range(class_num)]

    def solve(self, x, y):
        for i in range(self.class_num):
            print(f"Training classifier{i}...")
            self.classifiers[i].solve(x, self.re_labaling(y, i))
        return self

    def re_labaling(self, y, pos_label):
        """labelを受け取り、pos_labelに指定したカテゴリを+1、それ以外を-1にラベリングしなおしたデータを返す."""
        return np.where(y == pos_label, 1, -1)

    def argmax_by_E(self, result):

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if np.sum(result[:, j], axis=0) == -1: #case (1,-1,-1)
                    pass
                elif np.sum(result[:, j], axis=0) == 1: #case (1,1,-1)
                    a = np.array(np.where(result[:, j] == 1))[0][0]
                    b = np.array(np.where(result[:, j] == 1))[0][1]
                    if self.classifiers[a].energy > self.classifiers[b].energy:
                        result[a, j] = -1
                    else:
                        result[b, j] = -1
                elif np.sum(result[:, j], axis=0) == 3: #case (1,1,1)
                    min_e = np.argmin(np.array([self.classifiers[0].energy, self.classifiers[1].energy, self.classifiers[2].energy]))
                    result[0:min_e, j] = -1
                    result[min_e:i, j] = -1
                elif np.sum(result[:, j], axis=0) == -3: #case (-1,-1,-1)
                    min_e = np.argmin(
                        np.array([self.classifiers[0].energy, self.classifiers[1].energy, self.classifiers[2].energy]))
                    result[min_e, j] = 1

        #print("result shape after transform",result.shape)
        print("result", result)
        return np.argmax(result, axis=0)

    def predict(self, X):

        result = np.array([model.predict(X) for model in self.classifiers])
        #print("result", result)
        #print("result type and shape", type(result), result.shape)
        #result type and shape <class 'numpy.ndarray'> (3, 150)
        return self.argmax_by_E(result)

    def evaluate(self, X, y):
        pred = self.predict(X)
        print("pred result",pred)
        return accuracy_score(y, pred)

class OneVsOneClassifier():
    """一対一分類器."""

    def __init__(self, class_num, classifier, params=None):
        """binary classifierを受け取り、c(c-1)/2個のインスタンスを作成する."""
        self.class_num = class_num
        self.perm = list(combinations(list(range(class_num)), 2))
        if params is None:
            self.classifiers = [classifier() for _ in range(len(self.perm))]
        else:
            self.classifiers = [classifier(**params)
                                for _ in range(len(self.perm))]

    def fit(self, X, y):
        for i in range(len(self.classifiers)):
            X_i, y_i = self.extract_dataset(X, y, i)
            print(f"Training classifier{i}...")
            self.classifiers[i].fit(X_i, y_i)
        return self

    def extract_dataset(self, X, y, i):
        pos = self.perm[i][0]
        neg = self.perm[i][1]
        X = X[(y == pos) | (y == neg)]
        y = y[(y == pos) | (y == neg)]
        y = np.where(y == pos, 1, -1)
        return X, y

    def predict(self, X):
        votes = np.zeros((len(X), self.class_num))
        for i in range(len(self.classifiers)):
            prediction = self.classifiers[i].predict(X)
            voted = np.where(prediction == 1, self.perm[i][0], self.perm[i][1])
            for j in range(len(voted)):
                votes[j, voted[j]] += 1
        return np.argmax(votes, axis=1)

    def evaluate(self, X, y):
        pred = self.predict(X)
        return accuracy_score(y, pred)

class qSVM():
    def __init__(self, data, label, B=2, K=2, Xi=0, gamma=10, C=3, kernel="rbf", optimizer=SimulatedAnnealingOptimizer
                 ):
        """
        :param B:
        :param K:
        :param Xi:
        :param gamma:
        :param C: #still not used now
        :param kernel: default;rbf only rbf for now,
        :param optimizer:SA,DA,LASOLV
        """
        #self.data = data
        self.label = label
        self.B = B
        self.K = K
        self.N = data.shape[0]
        self.Xi = Xi
        self.gamma = float(gamma)
        self.C = C
        self.kernel = kernel

        self.options = {
            'SA': {}
        }
        self.optimizer = optimizer
        self.alpha = Array.create('alpha', shape=self.K * self.N, vartype='BINARY') #number of spins : K*N

        self.alpha_result = None
        self.alpha_result_array = None
        self.alpha_real = np.zeros((self.N,))

        self._support_vectors = None
        self._n_support = None
        self._alphas = None
        self._support_labels = None
        self._indices = None
        self.intercept = None
        self.energy = None




    def rbf(self, x, y):
        return np.exp(-1.0 * self.gamma * np.dot(np.subtract(x, y).T, np.subtract(x, y)))

    def transform(self, X):
        K = np.zeros([X.shape[0], X.shape[0]])
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                K[i, j] = self.rbf(X[i], X[j])
        return K

    def makeQUBO(self, data, label):

        alpha = self.alpha
        x = data
        t = label
        #t = self.label.astype(np.double)
        xi_p = Placeholder('Xi')

        energy = Sum(0, self.N, lambda n: Sum(0, self.N, lambda m: Sum(0,self.K, lambda k: Sum(0, self.K, lambda j:
                    alpha[self.K * n +k] * alpha[self.K * m +j] * t[n] * t[m] * self.rbf(x[n],x[m] * self.B ** (k+j))))))
        const_1 = Sum(0, self.N, lambda n: Sum(0, self.K, lambda k: alpha[self.K * n +k] * self.B ** k))
        const_2 = Constraint((Sum(0,self.N, lambda n: Sum(0, self.K, lambda k: alpha[self.K * n +k] * t[n] * self.B ** k)))
                             ** 2, label="alpha * t = 0")
        #const_3 = Sum(0, self.N, lambda n: Sum(0, self.K, lambda k: alpha[self.K * n + k] * self.B ** k) - self.C)

        h = 0.5 * energy - const_1 + xi_p * const_2
        #h = 0.5 * energy - const_1 + xi_p * const_2 + 50 * const_3

        model = h.compile()
        qubo, offset = model.to_qubo(feed_dict={'Xi': self.Xi})

        #print("qubo initial", qubo)

        return model,qubo

    def PyquboToNparray(self, qubo_dict):
        # qubo generate by pyqubo is tuple type data, this func for changing the qubo from dict to np.array
        # qubometasolver need a (spins_num,spins_num) size np.array type value
        # pyqubo generate a dict type qubo and only has the triangle of qubo like this
        # 2.9, 1.4, 2.6
        # null, 1.7, 5.2
        # null, null, 3.7
        # so dict length is spins*spins/2

        #print("Pyqubo before sort",qubo_dict)

        qubo = np.zeros((self.K * self.N, self.K * self.N))
        for i in range (self.K * self.N):
            for j in range (i, self.K * self.N):
                qubo[i][j] = qubo_dict.get(('alpha[%s]' % i, 'alpha[%s]' % j))
                if math.isnan(qubo[i][j]):
                    qubo[i][j] = qubo_dict.get(('alpha[%s]' % j, 'alpha[%s]' % i))

        #print(qubo)
        return qubo

    def solve(self, data, label):
        print("solving...")
        model, qubo = self.makeQUBO(data, label) #type(qubo) = dict
        print("Active optimizer: ", self.optimizer)

        if self.optimizer == SimulatedAnnealingOptimizer:
            sol = solve_qubo(qubo)  # solving by SA provided by neal #len(sol) = 80
            solution, broken, energy = model.decode_solution(sol, vartype="BINARY", feed_dict={'Xi': self.Xi})
            # sorted_function reture list
            sorted_solution = sorted(solution['alpha'].items(), key=lambda item: item[0], reverse=False)
            sorted_solution = np.array(sorted_solution)  # shape (80,2)
            # sorted_solution[:,0] is the index number/ sorted_solution[:,1]is binary value
            print("number of broken constarint = {}".format(len(broken)))
            self.energy = energy
            self.alpha_result = np.array(sorted_solution[:, 1])

        elif self.optimizer == DigitalAnnealerOptimizer:
            optimizer = self.optimizer(**self.options['DA'])
            q_matrix = self.PyquboToNparray(qubo)      # convert qubo type from dict to numpy array
            #print("q_matrix",q_matrix)
            #print("q_matrix",q_matrix.shape)
            solver = QUBOMetasolver(q_matrix, optimizer=optimizer)
            DA_result = solver.solve().get_best()
            # print(DA_result) #[{'qubo_spins': [0, 0, 1, 1, 1, 1, 0, 0], 'qubo_energy': -6}]
            # print("DA result type",type(DA_result)) #type: list
            solution = DA_result[0]['qubo_spins']
            energy = DA_result[0]['qubo_energy']
            self.alpha_result = np.array(solution)
            self.energy = energy
        else:
            print("This optimizer is not available")

        print("alpha_result = ", self.alpha_result)
        print("energy = {}".format(self.energy))

        K = self.transform(data)

        for i in range(self.N):
            for j in range(self.K):
                self.alpha_real[i] += self.alpha_result[self.K*i+j] * self.B ** j

        print("self.alpha_real", self.alpha_real)

        # print("(self.alpha_real)",self.alpha_real)
        is_sv = self.alpha_real > 1e-5
        # print("(self.alpha_real)", is_sv)
        self._support_vectors = data[is_sv]
        self._n_support = np.sum(is_sv)
        self._alphas = self.alpha_real[is_sv]
        self._support_labels = label[is_sv]
        self._indices = np.arange(data.shape[0])[is_sv]  # the index of supported vector
        self.intercept = 0

        for i in range(self._alphas.shape[0]):
            self.intercept += self._support_labels[i]
            self.intercept -= np.sum(self._alphas * self._support_labels * K[self._indices[i], is_sv])
        self.intercept /= self._alphas.shape[0]
        print("self.intercept", self.intercept)

        return self.alpha_result

    def signum(self, X):
        return np.where(X > 0, 1, -1)

    def predict(self, X):
        score = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            s = 0
            for alpha, label, sv in zip(self.alpha_result, self._support_labels, self._support_vectors):
                s += alpha * label * self.rbf(X[i], sv) * self.B ** self.K
            score[i] = s
        score = score + self.intercept
        return self.signum(score)

    def evaluate(self, X, y):
        pred = self.predict(X)
        return accuracy_score(y, pred)

