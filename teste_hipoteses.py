import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import json
from scikit_posthocs import posthoc_nemenyi_friedman
from IPython.display import display

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


    for metrica in ['f1', 'gmean']:
        print(f"\n### MÉTRICA: {metrica.upper()} ###")
        

        
        print("\nTeste de Friedman")
        statistic, p_value = stats.friedmanchisquare(
            df[df['loss_nome'] == 'binary_crossentropy'][metrica],
            df[df['loss_nome'] == 'focal_loss'][metrica],
            df[df['loss_nome'] == 'automatic_weighted_binary_crossentropy'][metrica]
        )

        df['id'] = np.repeat(np.arange(len(df) // 3), 3)

        if p_value < alpha:
            print("Há diferenças significativas entre pelo menos algumas das funções de loss.")
            
            # Teste post-hoc de Nemenyi para identificar quais grupos diferem
            try:
                posthoc_results = posthoc_nemenyi_friedman(
                    df.pivot_table(index='id', columns='loss_nome', values=metrica)
                )
                print("\nResultados do teste post-hoc de Nemenyi:")
                print(posthoc_results.to_string(float_format="{:.4f}".format))
                print("\nComparações significativas:")
                for i in range(len(posthoc_results)):
                    for j in range(i + 1, len(posthoc_results)):
                        if posthoc_results.iloc[i, j] < alpha:
                            print(f"{posthoc_results.index[i]} vs {posthoc_results.columns[j]}: p = {posthoc_results.iloc[i, j]:.4f}")
                            print(f"→ Existe evidência estatística de que {posthoc_results.index[i]} difere de {posthoc_results.columns[j]} em {metrica.upper()}.")

                            metric_1 = df[df['loss_nome'] == posthoc_results.index[i]][metrica].values.mean()
                            metric_2 = df[df['loss_nome'] == posthoc_results.columns[j]][metrica].values.mean()
                            print(f"  Média {posthoc_results.index[i]}: {metric_1:.4f}, Média {posthoc_results.columns[j]}: {metric_2:.4f}")
                            if metric_1 > metric_2:
                                print(f"  → {posthoc_results.index[i]} é melhor que {posthoc_results.columns[j]} em {metrica.upper()}.")
                            else:
                                print(f"  → {posthoc_results.columns[j]} é melhor que {posthoc_results.index[i]} em {metrica.upper()}.")

            except Exception as e:
                print(f"Erro ao executar teste post-hoc: {e}")
        else:
            print("Não há diferenças significativas entre as funções de loss.")

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


