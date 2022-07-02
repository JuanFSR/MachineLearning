import itertools
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score

from tqdm.notebook import tqdm

from joblib import delayed, Parallel

from scipy.stats import ttest_ind_from_stats

def calcular_estatisticas(resultados):
    return np.mean(resultados), np.std(resultados), np.min(resultados), np.max(resultados)

def imprimir_estatisticas(resultados):
    media, desvio, mini, maxi = calcular_estatisticas(resultados)
    print("Resultados: %.2f +- %.2f, min: %.2f, max: %.2f" % (media, desvio, mini, maxi))

def rejeitar_hip_nula(amostra1, amostra2, alpha=0.05):
    media_amostral_1, desvio_padrao_amostral_1, _, _ = calcular_estatisticas(amostra1)
    media_amostral_2, desvio_padrao_amostral_2, _, _ = calcular_estatisticas(amostra2)

    _, pvalor = ttest_ind_from_stats(media_amostral_1, desvio_padrao_amostral_1, len(amostra1), media_amostral_2, desvio_padrao_amostral_2, len(amostra2))
    return (pvalor <= alpha, pvalor)

def print_t_tests(resultados, cols=None, alpha=0.05):
    if cols is None:
        cols = sorted(resultados)    
    
    largura = max(max(map(len,cols))+2,12)
    
    print(" " * largura , end="")
    
    for t in cols:
        print(t.center(largura), end='')
    print()
    
    for t in sorted(resultados):
        print(t.center(largura), end='')
        for t2 in cols:
            d, p = rejeitar_hip_nula(resultados[t], resultados[t2], alpha=alpha)
            dif = '<' if np.mean(resultados[t]) - np.mean(resultados[t2]) < 0 else '>'
            print(("%.02f%s" % (p, (' (*%c)' % dif) if d else '')).center(largura), end='')
        print()
