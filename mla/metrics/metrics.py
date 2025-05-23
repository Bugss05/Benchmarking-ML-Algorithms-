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



def focal_loss_weighted(actual, predicted, EPS=1e-8, gamma=2.0, alpha=0.85):
    """
    Focal Loss binária com peso para classe positiva.

    Args:
        actual (array-like): rótulos reais (0 ou 1).
        predicted (array-like): probabilidades previstas p = P(y=1).
        EPS (float): valor pequeno para evitar log(0).
        gamma (float): parâmetro de focalização.
        alpha (float): peso para classe positiva (entre 0 e 1).

    Returns:
        float: valor médio da Focal Loss.
    """
    y = np.asarray(actual).astype(np.float32)
    p = np.clip(np.asarray(predicted), EPS, 1 - EPS)

    loss_pos = - alpha * (1 - p)**gamma * np.log(p) * y
    loss_neg = - (1 - alpha) * p**gamma * np.log(1 - p) * (1 - y)

    return np.mean(loss_pos + loss_neg)


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


def gradient_focal_loss_weighted(actual, predicted, EPS=1e-8, gamma=2.0, alpha=0.85):
    """
    Gradiente da Focal Loss binária ponderada com alpha.

    Args:
        actual: array-like de rótulos reais (0 ou 1).
        predicted: array-like de probabilidades previstas p = P(y=1).
        EPS: valor para evitar log(0) e divisões por 0.
        gamma: parâmetro de focalização.
        alpha: peso da classe positiva.

    Returns:
        grad: array-like com os gradientes dL/dp para cada exemplo.
    """
    y = np.asarray(actual).astype(np.float32)
    p = np.clip(np.asarray(predicted), EPS, 1 - EPS)

    # Derivada da focal loss para cada ponto
    grad_pos = -alpha * ((1 - p) ** gamma) * (gamma * np.log(p) / (1 - p) + 1 / p)
    grad_neg = -(1 - alpha) * (p ** gamma) * (gamma * np.log(1 - p) / p - 1 / (1 - p))

    grad = y * grad_pos + (1 - y) * grad_neg

    return grad / y.shape[0]  # média para manter compatível com loss.mean()

def gradient_categorical_crossentropy(actual, predicted):
    """
    Computes the gradient of the categorical cross-entropy loss with respect to the predictions.

    Args:
        actual (array-like): rótulos reais (0 ou 1).
        predicted (array-like): probabilidades previstas p = P(y=1).

    Returns:
        np.ndarray: Gradient of the loss, same shape as predicted.
    """
    return - (actual - predicted)

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
