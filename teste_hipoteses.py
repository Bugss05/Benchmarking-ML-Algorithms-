import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import json

def comparar_losses_metricas(df, alpha=0.05):
    def get_metricas(row):
        y_test = json.loads(row['test_out'])
        probs = json.loads(row['probs'])
        epoca = int(row['epocas']) - 1
        probs_epoca = probs[epoca][1]
        y_pred = [0 if p[0] > p[1] else 1 for p in probs_epoca]

        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        cm = confusion_matrix(y_test, y_pred).ravel()
        if len(cm) == 4:
            tn, fp, fn, tp = cm
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            esp = tn / (tn + fp) if (tn + fp) > 0 else 0
            gmean = np.sqrt(sens * esp)
        else:
            gmean = 0

        return f1, gmean

    df = df.copy()
    df[['f1', 'gmean']] = df.apply(lambda row: pd.Series(get_metricas(row)), axis=1)

    pares = [
        ("automatic_weighted_binary_crossentropy","binary_crossentropy"),
        ("automatic_weighted_binary_crossentropy", "focal_loss"),
        ("focal_loss","binary_crossentropy" )
    ]

    for metrica in ['f1', 'gmean']:
        print(f"\n### MÉTRICA: {metrica.upper()} ###")
        for l1, l2 in pares:
            m1 = df[df['loss_nome'] == l1][metrica].values
            m2 = df[df['loss_nome'] == l2][metrica].values

            if len(m1) != len(m2):
                print(f"\n Tamanhos diferentes: {l1} ({len(m1)}), {l2} ({len(m2)}). Ignorado.")
                continue

            # Teste unilateral: H1: m1 > m2
            stat, p = stats.wilcoxon(m1, m2, alternative='greater')
            conclusao = " MELHORIA SIGNIFICATIVA" if p < alpha else " NÃO HÁ MELHORIA SIGNIFICATIVA"

            print(f"\n{l1} > {l2} ({metrica}) via Wilcoxon | p = {p:.4f} → {conclusao}")
            if p < alpha:
                print(f"→ Existe evidência estatística de que {l1} supera {l2} em {metrica.upper()}.")
            else:
                print(f"→ Não se pode afirmar com confiança que {l1} supera {l2} em {metrica.upper()}.\n")

    # Histogramas
    for metrica in ['f1', 'gmean']:
        plt.figure(figsize=(10, 6))
        for loss in df['loss_nome'].unique():
            valores = df[df['loss_nome'] == loss][metrica].values
            plt.hist(valores, bins=10, alpha=0.5, label=loss)
        plt.title(f"Distribuição da métrica {metrica.upper()} por função de loss")
        plt.xlabel(metrica.upper())
        plt.ylabel("Frequência")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
