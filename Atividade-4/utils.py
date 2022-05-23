import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from sklearn.svm import SVC
import scipy.stats as stats
import itertools

def calcular_estatisticas(resultados):
    return np.mean(resultados), np.std(resultados), np.min(resultados), np.max(resultados)

def imprimir_estatisticas(resultados):
    med, std, mini, maxi = calcular_estatisticas(resultados)
    print("Resultados: %.2f +- %.2f, min: %.2f, max: %.2f" % (med, std, mini, maxi))

def rejeitar_hip_nula_testet(med_amostral1, std_amostral1, n1,
                         med_amostral2, std_amostral2, n2,
                         alpha=0.05):
    _, pvalor = stats.ttest_ind_from_stats(med_amostral1, std_amostral1, n1,
                                           med_amostral2, std_amostral2, n2)
    return pvalor, pvalor < alpha

def selecionar_melhor_svm(cs, gammas, X_treino, X_val, y_treino, y_val):
    
    def treinar_svm(c, gamma, X_treino, X_val, y_treino, y_val):
        svm = SVC(C=c, gamma=gamma, kernel='rbf')
        svm.fit(X_treino, y_treino)
        pred = svm.predict(X_val)
        return accuracy_score(y_val, pred)
    
    combinacoes = list(itertools.product(cs, gammas))
    acuracias_val = Parallel(n_jobs=4)(delayed(treinar_svm)
                            (c, g, X_treino, X_val, y_treino, y_val) 
                                       for c, g in combinacoes)       
        
    melhor_val = max(acuracias_val)
    melhor_comb = combinacoes[np.argmax(acuracias_val)]        
    svm = SVC(C=melhor_comb[0], gamma=melhor_comb[1], kernel='rbf')
    svm.fit(np.vstack((X_treino, X_val)), [*y_treino, *y_val])
    
    return svm, melhor_comb, melhor_val

def do_cv_svm(X, y, cv_splits, cs=[1], gammas=['auto']):

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    acuracias = []
    
    pgb = tqdm(total=cv_splits, desc='Folds avaliados')
    
    for treino_idx, teste_idx in skf.split(X, y):

        X_treino = X[treino_idx]
        y_treino = y[treino_idx]

        X_teste = X[teste_idx]
        y_teste = y[teste_idx]

        X_treino, X_val, y_treino, y_val = train_test_split(X_treino, y_treino, stratify=y_treino, test_size=0.2, random_state=1)

        ss = StandardScaler()
        ss.fit(X_treino)
        X_treino = ss.transform(X_treino)
        X_teste = ss.transform(X_teste)
        X_val = ss.transform(X_val)

        svm, _, _ = selecionar_melhor_svm(cs, gammas, X_treino, 
                                          X_val, y_treino, y_val)
        pred = svm.predict(X_teste)

        acuracias.append(accuracy_score(y_teste, pred))
        
        pgb.update(1)
        
    pgb.close()
    
    return acuracias

def selecionar_melhor_k_knn(ks, X_treino, X_val, y_treino, y_val):
    
    def treinar_knn(k, X_treino, X_val, y_treino, y_val):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_treino, y_treino)
        pred = knn.predict(X_val)
        return accuracy_score(y_val, pred)
        
    acuracias_val = Parallel(n_jobs=4)(delayed(treinar_knn)(k, X_treino, X_val, y_treino, y_val) for k in ks)       
        
    melhor_val = max(acuracias_val)
    melhor_k = ks[np.argmax(acuracias_val)]        
    knn = KNeighborsClassifier(n_neighbors=melhor_k)
    knn.fit(np.vstack((X_treino, X_val)), [*y_treino, *y_val])
    
    return knn, melhor_k, melhor_val

def do_cv_knn(X, y, cv_splits, ks):

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    acuracias = []
    classification_reports = []
    
    pgb = tqdm(total=cv_splits, desc='Folds avaliados')
    
    for treino_idx, teste_idx in skf.split(X, y):

        X_treino = X[treino_idx]
        y_treino = y[treino_idx]

        X_teste = X[teste_idx]
        y_teste = y[teste_idx]

        X_treino, X_val, y_treino, y_val = train_test_split(X_treino, y_treino, stratify=y_treino, test_size=0.2, random_state=1)

        ss = StandardScaler()
        ss.fit(X_treino)
        X_treino = ss.transform(X_treino)
        X_teste = ss.transform(X_teste)
        X_val = ss.transform(X_val)

        knn, _, _ = selecionar_melhor_k_knn(ks, X_treino, X_val, y_treino, y_val)
        pred = knn.predict(X_teste)

        acuracias.append(accuracy_score(y_teste, pred))
        
        pgb.update(1)
        
    pgb.close()
    
    return acuracias