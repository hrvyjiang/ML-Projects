# Implementation of SVM algorithm from scratch

from __future__ import division, print_function
import numpy as np
import cvxopt
from mlfromscratch.utils import train_test_split, normalize, accuracy_score
from mlfromscratch.utils.kernels import *
from mlfromscratch.utils import Plot

# Hide cvxopt output
cvxopt.solvers.options['show_progress'] = False

class SupportVenctorMachine(object):
    """
    Support vector machine classifier

    Parameters:
    C:float
        Penalty term
    kernel: function
        Kernel function. Can be either polynomial, rbf, or linear
    power: int
        degree of polunomial kernel. Will be ignored by the other kernel functions
    gamma: float
        Used in the rbf kernel function
    coef: float
        Bias term used in the polunomial kernel function
    """

    def __init__(self, C=1, kernel, power, gamma, coef):
        self.C = C
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.lagr_multiliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None

    def fit(self, X, y):
        
        n_samples, n_features = np.shape(X)

        # Set gamma to 1/n_features by default
        if not self.gamma:
            self.gamma = 1 / n_features

        # Initialize kernel method with parameters
        self.kernel = self.kernel(
            power=self.power,
            gamma=self.gamma,
            coef=self.coef)
        
        #Calculate kernel matrix
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])

        #Define the quadratic optimization problem
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tx='d')
        q = cvxopt.matrix(np.ones(n_samples ) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if not self.C:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))
        
        #Solve the quadratic optimization using cvxopt
        minimization = cvxopt.solvers(P, q, G, h, A, b)

        #Lagrange multipliers
        lagr_mult = np.ravel(minimization['x'])

        #Extract support vectors
        #Get indexes of non zero larg multipliers
        idx = lagr_mult > 1e-7
        #Get corresponding lagr multiplier
        self.lagr_multiliers - lagr_mult[idx]
        #Get samples that will act as support vectors
        self.support_vectors = X[idx]
        #Get corresponding label
        self.support_vector_labels = y[idx]

        #Calculate intercept with first support vector
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multiliers)):
            self.intercept -= self.lagr_multiliers[i] * self.support_vector_labels[i] * self.kernel(self.support_vectors[i], self.support_vectors[0])


    def predict(self, X):
        y_pred = []
        #Iterate through list of samples and predict
        for sample in X:
            prediction = 0
            #Determin the label of sample by the support vectors
            for i in range(len(self.lagr_multiliers)):
                prediction += self.lagr_multiliers[i] * self.support_vector_labels[i] * self.kernel(self.support_vectors[i], sample)
            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)
