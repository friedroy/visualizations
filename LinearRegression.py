import numpy as np
from typing import Callable


def polynomial_basis_functions(degree: int) -> Callable:
    """
    Create a function that calculates the polynomial basis functions up to (and including) a degree
    :param degree: the maximal degree of the polynomial basis functions
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             polynomial basis functions, a numpy array of shape [N, degree+1]
    """
    def pbf(x: np.ndarray):
        return np.concatenate([(x**i)[:, None]/degree for i in range(degree+1)], axis=1)
    return pbf


def gaussian_basis_functions(centers: np.ndarray, beta: float) -> Callable:
    """
    Create a function that calculates Gaussian basis functions around a set of centers
    :param centers: an array of centers used by the basis functions
    :param beta: a float depicting the lengthscale of the Gaussians
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             Gaussian basis functions, a numpy array of shape [N, len(centers)+1]
    """
    def gbf(x: np.ndarray):
        y = np.concatenate([np.exp(-.5*((x-m)**2)/(beta**2))[:, None] for m in centers], axis=1)
        return np.concatenate([np.ones((len(x), 1)), y], axis=1)
    return gbf


def sigmoid_basis_functions(centers: np.ndarray) -> Callable:
    """
    Create a function that calculates sigmoidal basis functions around a set of centers
    :param centers: an array of centers used by the basis functions
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             sigmoid basis functions, a numpy array of shape [N, len(centers)+1]
    """
    def sbf(x: np.ndarray):
        y = np.concatenate([1/(1+np.exp(-(x-m)))[:, None] for m in centers], axis=1)
        return np.concatenate([np.ones((len(x), 1)), y], axis=1)
    return sbf


def spline_basis_functions(knots: np.ndarray) -> Callable:
    """
    Create a function that calculates the cubic regression spline basis functions around a set of knots
    :param knots: an array of knots that should be used by the spline
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             cubic regression spline basis functions, a numpy array of shape [N, len(knots)+4]
    """
    def csbf(x: np.ndarray):
        x1 = np.concatenate([(x**i)[:, None] for i in range(4)], axis=1)
        x2 = np.concatenate([(x-k)[:, None] for k in knots], axis=1)
        x2[x2 < 0] = 0
        return np.concatenate([x1, x2**3], axis=1)
    return csbf


class BayesianLinearRegression:
    def __init__(self, theta_mean: np.ndarray, theta_cov: np.ndarray, sig: float, basis_functions: Callable):
        """
        Initializes a Bayesian linear regression model
        :param theta_mean:          the mean of the prior
        :param theta_cov:           the covariance of the prior
        :param sig:                 the signal noise to use when fitting the model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.mu = theta_mean
        self.cov = theta_cov
        self.log_det = np.linalg.slogdet(theta_cov)[-1]
        self.prec = np.linalg.inv(theta_cov)

        self.sig = sig
        self.phi = basis_functions
        self.trained = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        self.trained = True
        H = self.phi(X)
        self.mu = H.T@y[:, None]/self.sig + self.prec@self.mu[:, None]
        self.prec = self.prec + H.T@H/self.sig
        self.cov = np.linalg.inv(self.prec)
        self.mu = np.linalg.solve(self.prec, self.mu)[:, 0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model using MMSE
        :param X: the samples to predict
        :return: the predictions for X
        """
        return (self.phi(X)@self.mu[:, None])[:, 0]

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        return np.sqrt(np.diagonal(self.phi(X) @ self.cov @ self.phi(X).T) + self.sig)

    def posterior_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model and sampling from the posterior
        :param X: the samples to predict
        :return: the predictions for X
        """
        H = self.phi(X)
        chol = np.linalg.cholesky(self.cov.astype(np.float64) + np.eye(self.cov.shape[-1])*1e-10)
        theta = self.mu[:, None] + chol@np.random.randn(chol.shape[-1], 1)
        return (H@theta)[:, 0]

    def log_evidence(self, X: np.ndarray, y:np.ndarray) -> float:
        mu_0 = self.mu.copy()
        prec_0 = self.prec.copy()
        self.fit(X, y)
        H = self.phi(X)
        dist_y = -0.5*(np.sum((H@self.mu[:, None] - y[:, None])**2)/self.sig + len(y)*np.log(self.sig))
        mu_meaned = self.mu - mu_0
        dist_mu = -0.5*(np.sum(mu_meaned*(prec_0@mu_meaned[:, None])[:, None]) + self.log_det)
        return dist_y + dist_mu - 0.5*np.linalg.slogdet(self.prec)[-1]

    def evidence(self, X:np.ndarray, y:np.ndarray) -> float:
        return np.exp(self.log_evidence(X, y))


class LinearRegression:

    def __init__(self, basis_functions: Callable):
        """
        Initializes a linear regression model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.phi = basis_functions
        self.theta = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the model to the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        self.theta = (np.linalg.pinv(self.phi(X)) @ y[:, None])[:, 0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        assert self.theta is not None, 'Model must be trained to predict'
        return (self.phi(X) @ self.theta[:, None])[:, 0]

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the model and return the predicted values for X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)