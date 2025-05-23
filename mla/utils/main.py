# coding:utf-8
import numpy as np


def one_hot(y):
    n_values = np.max(y) + 1
    return np.eye(n_values)[y]

import numpy as np

def balanced_batch_iterator(X, y, batch_size=64, minority_class=1):
    """
    Gera batches de X e y contendo pelo menos uma amostra da classe minoritária por batch.
    """
    # Índices das classes
    idx_minor = np.where(y == minority_class)[0]
    idx_major = np.where(y != minority_class)[0]

    # Embaralhar os índices
    np.random.shuffle(idx_minor)
    np.random.shuffle(idx_major)

    # Calcular número de batches possíveis
    total_samples = len(y)
    num_batches = int(np.ceil(total_samples / batch_size))
    minor_i = 0
    major_i = 0

    for _ in range(num_batches):
        # Garantir 1 amostra da classe minoritária
        if minor_i >= len(idx_minor):
            # Reiniciar minoritários se acabarem
            np.random.shuffle(idx_minor)
            minor_i = 0
        idx_batch = [idx_minor[minor_i]]
        minor_i += 1

        # Preencher o restante com classe majoritária
        remaining = batch_size - 1
        if major_i + remaining > len(idx_major):
            # Reembaralhar majoritários se acabarem
            np.random.shuffle(idx_major)
            major_i = 0
        idx_batch += idx_major[major_i:major_i+remaining]
        major_i += remaining

        # Shuffle final do batch para não ficar sempre o minoritário em primeiro
        np.random.shuffle(idx_batch)

        yield X[idx_batch], y[idx_batch]


def batch_iterator(X,y=None, batch_size=64,equilibrar_batches=False):
    """Splits X into equal sized chunks."""
    if equilibrar_batches:
        balanced_batch_iterator(X, y, batch_size=batch_size)
    else:
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size
        batch_end = 0

        for b in range(n_batches):
            batch_begin = b * batch_size
            batch_end = batch_begin + batch_size

            X_batch = X[batch_begin:batch_end]

            yield X_batch

        if n_batches * batch_size < n_samples:
            yield X[batch_end:]
