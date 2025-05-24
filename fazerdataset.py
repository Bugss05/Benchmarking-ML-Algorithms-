import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import json
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix
from IPython.display import display  # se estiver num ambiente Jupyter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, recall_score

from sklearn.metrics import precision_score, recall_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from IPython.display import display_html


def adicionar_modelo_ao_dataset(model, dataset=None):
    """
    Adiciona os dados de um modelo ao dataset, com colunas simplificadas e listas como strings.
    
    Args:
        model: objeto com os atributos especificados.
        dataset: um DataFrame existente ou None para criar um novo.
    
    Returns:
        Um DataFrame com os dados atualizados.
    """
    nova_linha = {
        'ficheiro': model.name,
        'train_out': json.dumps(model.y_train),
        'test_out': json.dumps(model.y_test),
        'probs': json.dumps(model.metric_list),
        'epocas': model.max_epochs,
        'loss_list': json.dumps(model.loss_list),
        'loss_nome': model.loss_name,
        'l2': model.l2,
        'dropout': model.dropout,
        'zeros': model.zeros,
        'uns': model.uns
    }
    
    if dataset is None:
        dataset = pd.DataFrame([nova_linha])
    else:
        dataset = pd.concat([dataset, pd.DataFrame([nova_linha])], ignore_index=True)
    
    return dataset

def processar_modelos_subset(dataset,
                              roc_curve=False,
                              f1_score=False,
                              gmean=False,
                              confusion_matrix=False,
                              prec_recall_curve=False,
                              modo_visual='todos'):
    """
    Itera por todos os ficheiros únicos no dataset e chama a função de plotagem
    para cada conjunto de entradas com o mesmo nome de ficheiro.
    Apenas executa funções que operam por subset (linha a linha).
    """
    ficheiros_unicos = dataset['ficheiro'].unique()

    for nome_ficheiro in ficheiros_unicos:
        subset = dataset[dataset['ficheiro'] == nome_ficheiro]

        plotar_graficos(
            subset,
            roc_curve=roc_curve,
            f1_score=f1_score,
            gmean=gmean,
            confusion=confusion_matrix,
            prec_recall=prec_recall_curve,
            # Desativa os de dataset
            roc_vs_ratio=False,
            prec_rec_vs_ratio=False,
            modo_visual=modo_visual

        )


def plotar_graficos(dados, 
                               roc_curve=False, 
                               f1_score=False,
                               gmean=False,
                               confusion=False,
                               prec_recall=False,
                               roc_vs_ratio=False,
                               prec_rec_vs_ratio=False,
                               gmean_vs_ratio=False,
                               modo_ratio='weighted',
                               modo_visual='todos'):
    """
    Se for passado um caminho (str), carrega o dataset completo.
    Se for passado um DataFrame, trata como subset.
    
    Funções por linha: roc_curve, f1_score, gmean, confusion, prec_recall
    Funções por dataset: roc_vs_ratio, prec_rec_vs_ratio
    """

    # Carrega ou usa diretamente
    if isinstance(dados, str):
        dataset = pd.read_csv(dados)
        tipo = "dataset"
    elif isinstance(dados, pd.DataFrame):
        dataset = dados
        tipo = "subset"
    else:
        raise ValueError("Argumento 'dados' deve ser um path (str) ou um DataFrame.")

    # Validação cruzada
    subset_funcoes = [roc_curve, f1_score, gmean, confusion, prec_recall]
    dataset_funcoes = [roc_vs_ratio, prec_rec_vs_ratio]

    if tipo == "dataset" and any(subset_funcoes):
        raise ValueError("Só pode usar funções que operam o dataset inteiro (roc_vs_ratio, prec_rec_vs_ratio) ao passar o dataset completo.")

    if tipo == "subset" and any(dataset_funcoes):
        raise ValueError("Só pode usar funções por linha (roc_curve, f1_score, etc.) ao passar um subset.")

    # Subset (funções por linha)
    if tipo == "subset":
        if roc_curve:
            plot_roc_curve(dataset, modo=modo_visual)

        if f1_score:
            plot_f1_score(dataset, modo=modo_visual)

        if gmean:
            plot_gmean_score(dataset, modo=modo_visual)

        if confusion:
            plot_confusion_matrix(dataset, modo=modo_visual)

        if prec_recall:
            plot_precision_recall_curve_final(dataset, modo=modo_visual)

    # Dataset completo (funções comparativas)
    if tipo == "dataset":
        metrics = []
        if roc_vs_ratio:
            metrics.append('roc')
        if prec_rec_vs_ratio:
            metrics.extend(['precision', 'recall'])

        if gmean_vs_ratio:
            metrics.append('gmean')
        if metrics:
            plot_metrics_vs_imbalance(
                dataset,
                modo=modo_visual,
                x_metric=modo_ratio,
                metrics=tuple(set(metrics))
            )


def plot_roc_curve(subset, modo='todos'):
    """
    Gera as curvas ROC, com base no modo:
    - 'unico': Gera um único gráfico com todas as curvas ROC.
    - 'individual': Gera gráficos individuais para cada linha.
    - 'todos': Gera ambos, o gráfico único e os gráficos individuais.

    Args:
        subset: O subset dos dados que contém as informações de cada modelo.
        modo: O modo de plotagem ('unico', 'individual', ou 'todos').
    """
    if modo == 'juntos' or modo == 'todos':
        # Gerar um único gráfico com todas as curvas ROC
        plt.figure(figsize=(8, 6))
        
        for i, row in subset.iterrows():
                
            y_test = json.loads(row['test_out'])
            probs = json.loads(row['probs'])
            epocas = int(row['epocas'])
            # Aceder às probabilidades do conjunto de teste na última época
            y_test = np.array(y_test)

            # Acessar probabilidades da classe positiva (teste)
            prob_test = np.array(probs[epocas - 1][1])[:, 1]  # <--- Correção aqui
            # Calcular curva ROC e AUC
            fpr, tpr, _ = roc_curve(y_test, prob_test)
            roc_auc = auc(fpr, tpr)
            # Adicionar curva ao gráfico
            
            plt.plot(fpr, tpr, label=f"{row['loss_nome']} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], 'k--')  # Linha diagonal
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"{row['ficheiro']} - Curvas ROC")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if modo == 'cada' or modo == 'todos':
        # Gerar gráficos individuais para cada linha do subset
        for i, row in subset.iterrows():
            y_test = json.loads(row['test_out'])
            probs = json.loads(row['probs'])
            epocas = int(row['epocas'])

            # Aceder às probabilidades do conjunto de teste na última época
            prob_test = np.array(probs[epocas - 1][1])[:, 1]  # [1] = test
            y_test = np.array(y_test)

            if y_test.ndim == 2:  # If one-hot encoded, convert to 1D
                y_test = np.argmax(y_test, axis=1)
            # Calcular curva ROC e AUC
            fpr, tpr, _ = roc_curve(y_test, prob_test)
            roc_auc = auc(fpr, tpr)

            # Criar gráfico individual
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}, {row['loss_nome']}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f"ROC - {row['ficheiro']}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

def plot_f1_score(subset, modo="cada"):
    """
    Plota o F1 Score ponderado por época (treino vs teste).
    
    Parâmetros:
    - subset: DataFrame contendo os dados.
    - modo: "cada", "todos" ou "ambos"
    """
    def plot_cada():
        for i, row in subset.iterrows():

            y_train = json.loads(row['train_out'])
            y_test = json.loads(row['test_out'])
            probs = json.loads(row['probs'])
            epocas = int(row['epocas'])
            f1_train_weighted = []
            f1_test_weighted = []

            for epoch in range(epocas):
                y_train_pred = [0 if prob[0] > prob[1] else 1 for prob in probs[epoch][0]]
                y_test_pred = [0 if prob[0] > prob[1] else 1 for prob in probs[epoch][1]]
                f1_train_weighted.append(f1_score(y_train, y_train_pred, average='weighted'))
                f1_test_weighted.append(f1_score(y_test, y_test_pred, average='weighted'))

            plt.figure(figsize=(6, 5))
            plt.plot(range(1, epocas+1), f1_train_weighted, label=f'Test -{row["loss_nome"]}', marker='o')
            plt.plot(range(1, epocas+1), f1_test_weighted, label=f'Train -{row["loss_nome"]}', marker='x')
            plt.xlabel('Época')
            plt.ylabel('F1 Score')
            plt.title(f'F1 Score - {row["ficheiro"]}')
            plt.legend()
            plt.grid(True)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.show()


    def plot_juntos():
        plt.figure(figsize=(8, 6))
        for i, row in subset.iterrows():
            y_test = json.loads(row['test_out'])
            probs = json.loads(row['probs'])
            epocas = int(row['epocas'])
            f1_test_weighted = []
            for epoch in range(epocas):
                y_test_pred = [0 if prob[0] > prob[1] else 1 for prob in probs[epoch][1]]
                f1_test_weighted.append(f1_score(y_test, y_test_pred, average='weighted'))

            plt.plot(range(1, epocas+1), f1_test_weighted, label=row['loss_nome'])

        plt.xlabel('Época')
        plt.ylabel('F1 Score (Teste)')
        plt.title(f'F1 Score {row["ficheiro"]} - Todos os Modelos')
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if modo == "cada":
        plot_cada()
    elif modo == "juntos":
        plot_juntos()
    elif modo == "todos":
        plot_cada()
        plot_juntos()
    

def plot_gmean_score(subset, modo="cada"):
    """
    Plota o Gmean por época (treino vs teste).
    
    Parâmetros:
    - subset: DataFrame com os dados.
    - modo: "cada", "todos" ou "ambos"
    """
    def plot_cada():
        for i, row in subset.iterrows():
            y_train = json.loads(row['train_out'])
            y_test = json.loads(row['test_out'])
            probs = json.loads(row['probs'])
            epocas = int(row['epocas'])
            gmean_train = []
            gmean_test = []

            for epoch in range(epocas):
                y_train_pred = [0 if prob[0] > prob[1] else 1 for prob in probs[epoch][0]]
                y_test_pred = [0 if prob[0] > prob[1] else 1 for prob in probs[epoch][1]]
                recall_train = recall_score(y_train, y_train_pred, average=None)
                recall_test = recall_score(y_test, y_test_pred, average=None)
                gmean_train.append(np.sqrt(np.prod(recall_train)))
                gmean_test.append(np.sqrt(np.prod(recall_test)))

            plt.figure(figsize=(6, 5))
            plt.plot(range(1, epocas+1), gmean_train, label='Test', marker='o')
            plt.plot(range(1, epocas+1), gmean_test, label='Train', marker='x')
            plt.xlabel('Época')
            plt.ylabel('Gmean')
            plt.title(f'Gmean por Época - {row["ficheiro"]}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


    def plot_todos():
        plt.figure(figsize=(8, 6))
        for i, row in subset.iterrows():

            y_test = json.loads(row['test_out'])
            probs = json.loads(row['probs'])
            epocas = int(row['epocas'])
            gmean_test = []

            for epoch in range(epocas):
                y_test_pred = [0 if prob[0] > prob[1] else 1 for prob in probs[epoch][1]]
                recall_test = recall_score(y_test, y_test_pred, average=None)
                gmean_test.append(np.sqrt(np.prod(recall_test)))

            plt.plot(range(1, epocas+1), gmean_test, label=row['loss_nome'])

        plt.xlabel('Época')
        plt.ylabel('Gmean (Teste)')
        plt.title(f'Gmean por Época - {row["ficheiro"]}')
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if modo == "cada":
        plot_cada()
    elif modo == "juntos":
        plot_todos()
    elif modo == "todos":
        plot_cada()
        plot_todos()



def plot_confusion_matrix(subset, modo="cada"):
    """
    Plota a matriz de confusão da última época (apenas teste) para cada linha do subset.

    Parâmetros:
    - subset: DataFrame com os dados.
    - modo: "cada", "todos" ou "ambos"
    """
    def plot_cada():
        for i, row in subset.iterrows():
            y_test = json.loads(row['test_out'])
            probs = json.loads(row['probs'])
            epocas = int(row['epocas'])

            # Pegar as probabilidades da última época (teste)
            probs_ultima_epoca = probs[epocas - 1][1]
            y_pred = [0 if prob[0] > prob[1] else 1 for prob in probs_ultima_epoca]
            cm = confusion_matrix(y_test, y_pred)

            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.title(f'Matriz de Confusão - {row["loss_nome"]}')
            plt.grid(False)
            plt.tight_layout()
            plt.show()


    def plot_todos():
        fig, axes = plt.subplots(nrows=1, ncols=len(subset), figsize=(5 * len(subset), 4))
        if len(subset) == 1:
            axes = [axes]  # garantir que seja iterável

        for ax, (i, row) in zip(axes, subset.iterrows()):

            y_test = json.loads(row['test_out'])
            probs = json.loads(row['probs'])
            epocas = int(row['epocas'])
            probs_ultima_epoca = probs[epocas - 1][1]

            y_pred = [0 if prob[0] > prob[1] else 1 for prob in probs_ultima_epoca]

            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, colorbar=False)
            ax.set_title(f"{row['loss_nome']}")
            ax.grid(False)


        plt.suptitle(f"Matrizes de Confusão {row['ficheiro']} ", fontsize=14)
        plt.tight_layout()
        plt.show()

    if modo == "cada":
        plot_cada()
    elif modo == "juntos":
        plot_todos()
    elif modo == "todos":
        plot_cada()
        plot_todos()



def plot_precision_recall_curve_final(subset, modo="cada"):
    """
    Plota a Precision-Recall Curve da última época (teste) para cada linha do subset.

    Parâmetros:
    - subset: DataFrame com os dados.
    - modo: "cada", "todos" ou "ambos"
    """
    def plot_cada():
        for i, row in subset.iterrows():
            y_test = json.loads(row['test_out'])
            probs = json.loads(row['probs'])
            epocas = int(row['epocas'])

            # Pegamos apenas as probabilidades da classe positiva
            prob_pos = [p[1] for p in probs[epocas - 1][1]]
            precision, recall, _ = precision_recall_curve(y_test, prob_pos)
            disp = PrecisionRecallDisplay(precision=precision, recall=recall)
            disp.plot()

            plt.title(f'Precision-Recall Curve - {row["ficheiro"]}')
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def plot_todos():
        plt.figure(figsize=(8, 6))
        for i, row in subset.iterrows():
            y_test = json.loads(row['test_out'])
            probs = json.loads(row['probs'])
            epocas = int(row['epocas'])
            prob_pos = [p[1] for p in probs[epocas - 1][1]]
            precision, recall, _ = precision_recall_curve(y_test, prob_pos)

            # Usando PrecisionRecallDisplay para cada modelo
            disp = PrecisionRecallDisplay(precision=precision, recall=recall)
            disp.plot(ax=plt.gca(), name=row['loss_nome'])

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - Todos os Modelos')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if modo == "cada":
        plot_cada()
    elif modo == "juntos":
        plot_todos()
    elif modo == "todos":
        plot_cada()
        plot_todos()



def plot_metrics_vs_imbalance(dataset, 
                              modo='all', 
                              x_metric='weighted',
                              metrics=('roc', 'precision', 'recall', 'gmean')):
    """
    Compara várias métricas (ROC AUC, Precision, Recall, G-mean) da última época (teste)
    para cada loss em função do desequilíbrio de classes.

    Args:
        dataset (pd.DataFrame): deve conter colunas
            - 'loss_nome'
            - 'test_out'  (string JSON de y_test)
            - 'probs'     (string JSON de probabilidades por época)
            - 'zeros', 'uns'
            - 'epocas'
        modo (str): 'each' → um scatter por loss, 
                    'all'  → todos losses num só scatter, 
                    'both' → faz os dois
        x_metric (str): 
            - 'ratio'   → x = minor/major  
            - 'weighted'→ x = (minor/major) * total_samples  
        metrics (tuple): métricas a plotar, subset de {'roc','precision','recall','gmean'}
    """
    grupos = dataset.groupby('loss_nome')
    data = {}

    for loss, sub in grupos:
        xs = []
        ys = {m: [] for m in metrics}
        for _, row in sub.iterrows():
            y_test = json.loads(row['test_out'])
            probs = json.loads(row['probs'])
            ep = int(row['epocas']) - 1
            p_pos = [p[1] for p in probs[ep][1]]
            y_pred = [1 if p > 0.5 else 0 for p in p_pos]

            if 'roc' in metrics:
                fpr, tpr, _ = roc_curve(y_test, p_pos)
                ys['roc'].append(auc(fpr, tpr))
            if 'precision' in metrics:
                ys['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            if 'recall' in metrics:
                ys['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            if 'gmean' in metrics:
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                gmean = np.sqrt(sensitivity * specificity)
                ys['gmean'].append(gmean)

            n0, n1 = row['zeros'], row['uns']
            minor, major = (n1, n0) if n1 < n0 else (n0, n1)
            ratio = minor / major if major > 0 else 0
            total = n0 + n1
            x = ratio * total if x_metric == 'weighted' else ratio
            xs.append(x)

        data[loss] = {'x': xs, **ys}

    def _plot_all(metric):
        plt.figure(figsize=(10, 8))
        for loss, vals in data.items():
            plt.scatter(vals['x'], vals[metric], label=loss,s=30)
        plt.axvline(x=40, color='red', linestyle='--', label='Equilíbrio (minor/major = 40)')
        plt.xlabel("Desequilíbrio (minor/major)" + (" × total" if x_metric == 'weighted' else ""))
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} vs Desequilíbrio por Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _plot_each(metric):
        for loss, vals in data.items():
            plt.figure(figsize=(6, 5))
            plt.scatter(vals['x'], vals[metric])
            plt.xlabel("Desequilíbrio (minor/major)" + (" × total" if x_metric == 'weighted' else ""))
            plt.ylabel(metric.upper())
            plt.axvline(x=40, color='red', linestyle='--', label='Equilíbrio (minor/major = 40)')
            plt.title(f"{metric.upper()} vs Desequilíbrio\nLoss: {loss}")
            plt.grid(True)
            plt.xlim(0, 800) 
            plt.tight_layout()
            plt.show()
            

    for metric in metrics:
        if modo in ('all', 'both', 'todos'):
            _plot_all(metric)
        if modo in ('each', 'both', 'todos'):
            _plot_each(metric)

def gerar_tabelas_percentuais_metricas(dataset, limiares=(0.5, 0.7, 0.8, 0.9), min_test_size=100):
    """
    Gera e exibe tabelas lado a lado com o NÚMERO de execuções acima dos limiares
    para as métricas: ROC AUC, Precision, Recall, Gmean. Cada tabela representa
    um tipo de loss (loss_nome). Só considera execuções com y_test >= min_test_size.

    Args:
        dataset (str ou pd.DataFrame): Caminho para CSV ou DataFrame.
        limiares (tuple): Limiares de avaliação para as métricas.
        min_test_size (int): Número mínimo de amostras em y_test para ser incluído.
    """
    tabelas_html = []

    for perda in dataset['loss_nome'].unique():
        sub = dataset[dataset['loss_nome'] == perda]

        valores_metricas = {'ROC AUC': [], 'Precision': [], 'Recall': [], 'Gmean': []}

        for _, row in sub.iterrows():
            y_test = json.loads(row['test_out'])

            if len(y_test) < min_test_size:
                continue

            probs = json.loads(row['probs'])
            ep = int(row['epocas']) - 1
            prob_pos = [p[1] for p in probs[ep][1]]
            y_pred = [1 if p > 0.5 else 0 for p in prob_pos] # Limiar de 0.5 para classificação binária

            try:
                roc = roc_auc_score(y_test, prob_pos)
            except ValueError: # Adicionado para tratar casos onde roc_auc_score não pode ser calculado
                roc = 0.0

            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)

            cm = confusion_matrix(y_test, y_pred).ravel()
            if len(cm) == 4:
                tn, fp, fn, tp = cm
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                gmean = np.sqrt(sensitivity * specificity) if (sensitivity * specificity) >= 0 else 0 # Garantir não raiz de negativo
            else: # Caso não seja possível calcular a matriz de confusão completa (e.g., só uma classe predita)
                gmean = 0.0

            valores_metricas['ROC AUC'].append(roc)
            valores_metricas['Precision'].append(prec)
            valores_metricas['Recall'].append(rec)
            valores_metricas['Gmean'].append(gmean)

        if not any(valores_metricas.values()): # Verifica se alguma lista de métricas tem valores
            continue

        if len(valores_metricas['ROC AUC']) == 0 and \
           len(valores_metricas['Precision']) == 0 and \
           len(valores_metricas['Recall']) == 0 and \
           len(valores_metricas['Gmean']) == 0:
            continue


        tabela = pd.DataFrame(index=list(valores_metricas.keys())) # Garante a ordem das métricas
        for limiar_valor in limiares: # Renomeado para clareza, ex: 0.5, 0.7
            coluna_contagens = []
            for metrica_nome in tabela.index:
                valores_da_metrica = np.array(valores_metricas[metrica_nome])
                if len(valores_da_metrica) == 0: # Se não houver valores para esta métrica
                    contagem = 0
                else:
                    contagem = np.sum(valores_da_metrica >= limiar_valor)
                coluna_contagens.append(contagem)
            tabela[f'{int(limiar_valor * 100)}%'] = coluna_contagens

        html = f'<h3 style="text-align:center">{perda}</h3>' + tabela.to_html(escape=False)
        tabelas_html.append(html)

    if tabelas_html:
        display_html(''.join(
            [f'<div style="display:inline-block; margin-right:20px; vertical-align:top;">{t}</div>' for t in tabelas_html]
        ), raw=True)
    else:
        print(f"Nenhuma 'loss_nome' com pelo menos {min_test_size} amostras em y_test ou sem dados de métricas válidos.")


