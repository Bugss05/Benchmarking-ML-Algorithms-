# coding:utf-8
import numpy as np


class BaseEstimator:
    y_required = True
    fit_required = True

    def _setup_input(self, X, y=None, X_test=None, Y_test=None):
        """Ensure inputs to an estimator are in the expected format.
    
        Parameters
        ----------
        X : array-like
            Feature dataset (train).
        y : array-like
            Target values for train.
        X_test : array-like, optional
            Feature dataset for testing.
        Y_test : array-like, optional
            Target values for testing.
        """
        
        # Setup for training data
        if not isinstance(X, np.ndarray):
            X = np.array(X)
    
        if X.size == 0:
            raise ValueError("Got an empty matrix for X.")
    
        if X.ndim == 1:
            self.n_samples, self.n_features = 1, X.shape
        else:
            self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])
    
        self.X = X
    
        if self.y_required:
            if y is None:
                raise ValueError("Missed required argument y")
    
            if not isinstance(y, np.ndarray):
                y = np.array(y)
    
            if y.size == 0:
                raise ValueError("The targets array must be non-empty.")
    
            self.y = y
    
        # Setup for testing data (optional)
        if X_test is not None:
            if not isinstance(X_test, np.ndarray):
                X_test = np.array(X_test)
            
            if X_test.size == 0:
                raise ValueError("Got an empty matrix for X_test.")
    
            self.X_test = X_test
    
            if Y_test is not None:
                if not isinstance(Y_test, np.ndarray):
                    Y_test = np.array(Y_test)
    
                if Y_test.size == 0:
                    raise ValueError("The targets array Y_test must be non-empty.")
    
                self.Y_test = Y_test
            else:
                raise ValueError("Provided X_test but missing corresponding Y_test.")
    
        else:
            self.X_test = None
            self.Y_test = None
        
    
    def fit(self, X, y=None):
        self._setup_input(X, y)

    def predict(self, X=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if self.X is not None or not self.fit_required:
            return self._predict(X)
        else:
            raise ValueError("You must call `fit` before `predict`")

    def _predict(self, X=None):
        raise NotImplementedError()
