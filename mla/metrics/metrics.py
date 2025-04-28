# coding:utf-8
import autograd.numpy as np
from sklearn.metrics import precision_recall_curve, recall_score as sk_recall_score, f1_score as sk_f1_score # ⬅️ Importação adicionada
EPS = 1e-15


def unhot(function):
    """Convert one-hot representation into one column."""

    def wrapper(actual, predicted):
        if len(actual.shape) > 1 and actual.shape[1] > 1:
            actual = actual.argmax(axis=1)
        if len(predicted.shape) > 1 and predicted.shape[1] > 1:
            predicted = predicted.argmax(axis=1)
        return function(actual, predicted)

    return wrapper


def absolute_error(actual, predicted):
    return np.abs(actual - predicted)


@unhot
def classification_error(actual, predicted):
    return (actual != predicted).sum() / float(actual.shape[0])


@unhot
def accuracy(actual, predicted):
    return 1.0 - classification_error(actual, predicted)


def mean_absolute_error(actual, predicted):
    return np.mean(absolute_error(actual, predicted))


def squared_error(actual, predicted):
    return (actual - predicted) ** 2


def squared_log_error(actual, predicted):
    return (np.log(np.array(actual) + 1) - np.log(np.array(predicted) + 1)) ** 2


def mean_squared_log_error(actual, predicted):
    return np.mean(squared_log_error(actual, predicted))


def mean_squared_error(actual, predicted):
    return np.mean(squared_error(actual, predicted))


def root_mean_squared_error(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))


def root_mean_squared_log_error(actual, predicted):
    return np.sqrt(mean_squared_log_error(actual, predicted))


def logloss(actual, predicted):
    predicted = np.clip(predicted, EPS, 1 - EPS)
    loss = -np.sum(actual * np.log(predicted))
    return loss / float(actual.shape[0])


def hinge(actual, predicted):
    return np.mean(np.max(1.0 - actual * predicted, 0.0))


def binary_crossentropy(actual, predicted):
    predicted = np.clip(predicted, EPS, 1 - EPS)
    return np.mean(-np.sum(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)))

def automatic_weighted_binary_crossentropy(actual, predicted, num_zeros, num_ones, EPS=1e-8):
    """
    Função de Binary Cross-Entropy com pesos automáticos ajustados para classes desbalanceadas,
    onde os números de 0's e 1's são passados como argumentos.

    Args:
    - actual (array-like): Lista ou vetor de rótulos reais (0 ou 1).
    - predicted (array-like): Lista ou vetor de probabilidades previstas para a classe positiva.
    - num_zeros (int): Número de ocorrências da classe 0.
    - num_ones (int): Número de ocorrências da classe 1.
    - EPS (float): Pequeno valor para evitar log(0).

    Returns:
    - loss (float): Valor da função de perda ajustada.
    """
    # Garantir que predicted esteja dentro do intervalo [EPS, 1 - EPS]
    predicted = np.clip(predicted, EPS, 1 - EPS)

    # Calcular o total de exemplos
    total_count = num_zeros + num_ones
    
    # Calcular o peso das classes (inverso da frequência da classe)
    weight_0 = total_count / (2.0 * num_zeros)  # Peso para a classe 0 (majoritária)
    weight_1 = total_count / (2.0 * num_ones)  # Peso para a classe 1 (minoritária)

    # Calculando a Binary Cross-Entropy com pesos
    loss = - (weight_0 * (actual == 0) * np.log(1 - predicted) + 
              weight_1 * (actual == 1) * np.log(predicted))
    
    return np.mean(loss)


def f1score(actual, predicted, threshold=None):
    """Compute the F1 score with optional threshold optimization."""
    EPS = 1e-8  # Para evitar divisões por zero

    # Converter `actual` de one-hot encoding para rótulos de classe
    if len(actual.shape) > 1 and actual.shape[1] > 1:
        actual = np.argmax(actual, axis=1)

    # Se predicted for uma matriz (ex: scores para várias classes), transforma
    if len(predicted.shape) > 1 and predicted.shape[1] > 1:
        predicted = np.argmax(predicted, axis=1)

    # Se predicted são scores contínuos entre 0 e 1, encontrar melhor threshold
    if predicted.dtype.kind in 'fc' or (predicted.max() > 1 or predicted.min() < 0):
        # Parece que são scores "sujos", já maiores que 1
        # Assume que são classes já (não scores)
        return sk_f1_score(actual, predicted, average="weighted")

    if (predicted.max() <= 1 and predicted.min() >= 0) and (len(np.unique(predicted)) > 2):
        if threshold is None:
            # Encontrar o melhor threshold automaticamente
            precisions, recalls, thresholds = precision_recall_curve(actual, predicted)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + EPS)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
        else:
            # Usar o threshold fornecido
            best_threshold = threshold
        
        # Aplicar o threshold para converter scores em classes
        predicted_labels = (predicted >= best_threshold).astype(int)
        
        return sk_f1_score(actual, predicted_labels, average="weighted"),best_threshold

    best_threshold=threshold
    # Se já são labels 0/1, calcula normal
    return sk_f1_score(actual, predicted, average="weighted"), best_threshold


def recall(actual, predicted, threshold=None):
    """Compute the Recall with optional threshold optimization."""
    EPS = 1e-8  # Para evitar divisões por zero

    # Converter `actual` de one-hot encoding para rótulos de classe
    if len(actual.shape) > 1 and actual.shape[1] > 1:
        actual = np.argmax(actual, axis=1)

    # Se predicted for uma matriz (ex: scores para várias classes), transforma
    if len(predicted.shape) > 1 and predicted.shape[1] > 1:
        predicted = np.argmax(predicted, axis=1)

    # Se predicted são scores contínuos entre 0 e 1, encontrar melhor threshold
    if predicted.dtype.kind in 'fc' or (predicted.max() > 1 or predicted.min() < 0):
        # Parece que são scores "sujos", já maiores que 1
        # Assume que são classes já (não scores)
        return sk_recall_score(actual, predicted, average="weighted")

    if (predicted.max() <= 1 and predicted.min() >= 0) and (len(np.unique(predicted)) > 2):
        if threshold is None:
            # Encontrar o melhor threshold automaticamente
            precisions, recalls, thresholds = precision_recall_curve(actual, predicted)
            best_idx = np.argmax(recalls)  # Melhor threshold para recall
            best_threshold = thresholds[best_idx]
        else:
            # Usar o threshold fornecido
            best_threshold = threshold
        
        # Aplicar o threshold para converter scores em classes
        predicted_labels = (predicted >= best_threshold).astype(int)
        
        return sk_recall_score(actual, predicted_labels, average="weighted"), best_threshold

    best_threshold = threshold
    # Se já são labels 0/1, calcula normal
    return sk_recall_score(actual, predicted, average="weighted"), best_threshold

# aliases
mse = mean_squared_error
rmse = root_mean_squared_error
mae = mean_absolute_error

def get_metric(name):
    """Return metric function by name"""
    try:
        return globals()[name]
    except Exception:
        raise ValueError("Invalid metric function.")
