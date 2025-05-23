import logging

import numpy as np
from autograd import elementwise_grad

from mla.base import BaseEstimator
from mla.metrics.metrics import get_metric
from mla.neuralnet.layers import PhaseMixin
from mla.neuralnet.loss import get_loss
from mla.neuralnet.loss_gradient import get_gradient_loss
from mla.utils import batch_iterator

np.random.seed(9999)

"""
Architecture inspired from:

    https://github.com/fchollet/keras
    https://github.com/andersbll/deeppy
"""


def gradient_automatic_weighted_bce(actual, predicted, EPS: float = 1e-8, pos_weight: float = 1.0) -> np.ndarray:
    """
    Calcula o gradiente da perda Binary Cross-Entropy (BCE) ponderada automáticamente,
    aceitando saída sigmoid com um ou dois neurônios (probabilidades de classe).

    Se `predicted` tem shape (N,) assume como probabilidades p1 (classe positiva).
    Se `predicted` tem shape (N,2) assume [p0,p1] de duas sigmoids independentes.

    Args:
        actual: array-like rótulos 0/1 ou one-hot (N,2).
        predicted: array-like probabilidades p1 (N,) ou [p0,p1] (N,2).
        EPS: valor para clipping de probabilidades.
        pos_weight: fator >1 para aumentar peso da classe positiva.

    Returns:
        np.ndarray gradientes dL/dp com mesma forma de `predicted`.

    Raises:
        ValueError: se shapes de `actual` e `predicted` não coincidirem em N.
    """
    # Converter para numpy
    actual_arr = np.asarray(actual)
    pred_arr = np.asarray(predicted)

    # Extrair labels
    if actual_arr.ndim > 1 and actual_arr.shape[1] == 2:
        labels = np.argmax(actual_arr, axis=1)
    else:
        labels = actual_arr.flatten().astype(int)
    N = labels.size

    # Processar diferentes formatos de predicted
    if pred_arr.ndim == 2 and pred_arr.shape[1] == 2:
        # Duas sigmoids independentes: col0 = p0, col1 = p1
        p0 = pred_arr[:, 0].flatten()
        p1 = pred_arr[:, 1].flatten()
        if p0.size != N or p1.size != N:
            raise ValueError(f"Shape mismatch: actual {N}, predicted {pred_arr.shape}")
        # Clipping
        p0 = np.clip(p0, EPS, 1 - EPS)
        p1 = np.clip(p1, EPS, 1 - EPS)
    else:
        # Vetor de probabilidades p1
        p1 = pred_arr.flatten()
        if p1.size != N:
            raise ValueError(f"Shape mismatch: actual {N}, predicted {p1.size}")
        p1 = np.clip(p1, EPS, 1 - EPS)
        p0 = 1 - p1

    # Contagens de classes
    n0 = int(np.sum(labels == 0)) or 1
    n1 = int(np.sum(labels == 1)) or 1

    # Pesos
    w0 = N / (2.0 * n0)
    w1 = (N / (2.0 * n1)) * pos_weight

    # Gradientes BCE ponderados
    # para p1: dL/dp1 = [-(y * w1)/p1 + ((1-y) * w0)/p0] / N
    grad_p1 = (-(labels * w1) / p1 + ((1 - labels) * w0) / p0) / N
    # para p0: dL/dp0 = - (grad_p1)  (derivada simétrica)
    grad_p0 = -grad_p1

    # Montar saída
    if pred_arr.ndim == 2 and pred_arr.shape[1] == 2:
        grad = np.stack([grad_p0, grad_p1], axis=1)
    else:
        grad = grad_p1

    return grad




class NeuralNet(BaseEstimator):
    fit_required = False

    def __init__(
        self , layers , filename , optimizer , loss , l2 ,
         dropout , count_ones , count_zeros ,
        max_epochs=10 , batch_size=64 , metric="mse",
        shuffle = False , verbose=True , testarerros = False,
        zeros=0, uns=0 , equilibrar_batches = False
     ):
        self.verbose = verbose
        self.shuffle = shuffle
        self.optimizer = optimizer

        self.loss = get_loss(loss)
        self.loss_name = loss
        self.loss_grad = get_gradient_loss(loss)
        self.error_list= []
        self.metric_list= []
        self.loss_history= []
        self.testar_differros=testarerros
        self.metric = get_metric(metric)
        self.layers = layers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self._n_layers = 0
        self.log_metric = True if loss != metric else False
        self.metric_name = metric
        self.bprop_entry = self._find_bprop_entry()
        self.training = False
        self._initialized = False
        self.uns=uns
        self.zeros=zeros
        self.unst=count_ones
        self.zerost=count_zeros
        self.l2= l2
        self.dropout=dropout
        self.equilibrar_batches = False
        self.filename=filename



    def _setup_layers(self, x_shape):
        """Initialize model's layers."""
        x_shape = list(x_shape)
        x_shape[0] = self.batch_size

        for layer in self.layers:
            layer.setup(x_shape)
            x_shape = layer.shape(x_shape)

        self._n_layers = len(self.layers)
        # Setup optimizer
        self.optimizer.setup(self)
        self._initialized = True
        #logging.info("Total parameters: %s" % self.n_params)

    def _find_bprop_entry(self):
        """Find entry layer for back propagation."""

        if len(self.layers) > 0 and not hasattr(self.layers[-1], "parameters"):
            return -1
        return len(self.layers)

    def fit(self, X, y=None,xtest=None,ytest=None):
        
        if not self._initialized:
            self._setup_layers(X.shape)
        
        if y.ndim == 1:
            # Reshape vector to matrix
            y = y[:, np.newaxis]
        if xtest is not None or ytest is not None:
            self._setup_input(X, y, xtest, ytest)
        else:
            self._setup_input(X, y)

        self.is_training = True
        # Pass neural network instance to an optimizer
        self.optimizer.optimize(self)
        self.is_training = False

    def update(self, X, y):
        # Forward pass
        y_pred = self.fprop(X)

        # Backward pass
        grad = self.loss_grad(y, y_pred)
        for layer in reversed(self.layers[: self.bprop_entry]):
            grad = layer.backward_pass(grad)

        if self.loss_name == "automatic_weighted_binary_crossentropy":
            return self.loss(y, y_pred,self.zeros,self.uns)
        else : return self.loss(y, y_pred)

    def fprop(self, X):
        """Forward propagation."""
        for layer in self.layers:
            X = layer.forward_pass(X)
        return X

    def _predict(self, X=None):
        if not self._initialized or (self.testar_differros and self._initialized):
            self._setup_layers(X.shape)

        y = []
        y.append(self.fprop(X))
        return np.concatenate(y)

    @property
    def parametric_layers(self):
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                yield layer

    @property
    def parameters(self):
        """Returns a list of all parameters."""
        params = []
        for layer in self.parametric_layers:
            params.append(layer.parameters)
        return params

    def error(self, X=None, y=None,testar=False):
        """Calculate an error for given examples."""
        training_phase = self.is_training
        if training_phase:
            # Temporally disable training.
            # Some layers work differently while training (e.g. Dropout).
            self.is_training = False
        if X is None and y is None:
            y_pred = self._predict(self.X)
            trainscore,th = self.metric(self.y, y_pred)
        else:
            y_pred = self._predict(X)
            trainscore ,th= self.metric(y, y_pred)
            
        if testar:
            y_pred = self._predict(self.X_test)
            testscore ,thl= self.metric(self.Y_test, y_pred,th)
            self.error_list.append((trainscore, testscore))

        if training_phase:
            self.is_training = True
        return trainscore

    @property
    def is_training(self):
        return self.training

    @is_training.setter
    def is_training(self, train):
        self.training = train
        for layer in self.layers:
            if isinstance(layer, PhaseMixin):
                layer.is_training = train

    def shuffle_dataset(self):
        """Shuffle rows in the dataset."""
        n_samples = self.X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        self.X = self.X.take(indices, axis=0)
        self.y = self.y.take(indices, axis=0)

    @property
    def n_layers(self):
        """Returns the number of layers."""
        return self._n_layers

    @property
    def n_params(self):
        """Return the number of trainable parameters."""
        return sum([layer.parameters.n_params for layer in self.parametric_layers])

    def reset(self):
        self._initialized = False
