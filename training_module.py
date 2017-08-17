import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib
import pickle
import warnings
import pdb

# funcion para elegir el umbral que resulte en valores balanceados en la diagonal    
def get_optimal_thr_diagonal_cm(probs, target, step): 
    difference = np.zeros((len(np.arange(0,1,step))))
    n=-1
    for thr in np.arange(0,1,step):
        preds_thresholded = np.zeros(len(probs))
        n=n+1
        preds_thresholded[np.where(probs>thr)[0]] = 1
        cm = confusion_matrix(target, preds_thresholded).astype(float)
        cm[0,:] = cm[0,:]/float(sum(cm[0,:]))
        cm[1,:] = cm[1,:]/float(sum(cm[1,:]))
        difference[n] = abs(cm[0,0] - cm[1,1])
    loc = np.where( difference==min(difference))[0]
    return np.arange(0,1,step)[loc][0]
    
# funcion para expandir las matrices en una lista y tomar la parte triangular superior    
def unfold_data(data_list): 
    output = np.zeros((len(data_list), len(data_list[0][np.triu_indices(data_list[0].shape[0],1)])))
    for i,matrix in enumerate(data_list):
        output[i,:] = matrix[np.triu_indices(data_list[0].shape[0],1)]
    return output

def train_set(clf_type,shuffle=False):
        
    n_estimators = 100 # cantidad de arboles
    n_folds = 5 # cantidad de folds

    # cargar matrices de correlacion
    with open('datasets/slee_data.pickle', "rb") as input_file:
       sleep_data = pickle.load(input_file)
       
    set1 = unfold_data(sleep_data['W'])  # seleccionar que par de fases de suenio se van a comparar
    set2 = unfold_data(sleep_data['N3'])

    target1 = np.zeros(set1.shape[0])
    target2 = np.ones(set2.shape[0])
    	
    target = np.concatenate((target1, target2), axis=0)

    # para clasificadores random
    if shuffle:
        np.random.shuffle(target)

    data = np.concatenate((set1,set2), axis=0)

    cv = StratifiedKFold(target, n_folds=n_folds) # crear objeto de cross validation estratificada
        
    cv_target = np.array([])
    cv_prediction = np.array([])
    cv_probas = np.array([])
    cv_importances = np.zeros((n_folds, data.shape[1] ))

    feature_imp=np.zeros((set1.shape[1],n_folds))
        
    for i, (train, test) in enumerate(cv):
            
        X_train = data[train] # crear sets de entrenamiento y testeo para el fold
        X_test = data[test]
        y_train = target[train]
        y_test = target[test]
        
        if clf_type == 'rf2':
            clf = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=2)
        if clf_type == 'rf5':
            clf = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=5)
        if clf_type == 'rf5oob':
            clf = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=5, oob_score=True)

        if clf_type == 'svc':
            clf = svm.SVC(probability=True)
        if clf_type == 'nusvc':
            clf = svm.NuSVC(nu=0.1,probability=True)
        if clf_type == 'polysvc':
            clf = svm.SVC(probability=True, kernel='poly')
        if clf_type == 'sigsvc':
            clf = svm.SVC(probability=True, kernel='sigmoid')
        if clf_type == 'polysvc5':
            clf = svm.SVC(probability=True, kernel='poly', degree=5)
        if clf_type == 'polysvc2':
            clf = svm.SVC(probability=True, kernel='poly', degree=2)
        if clf_type == 'polysvc1':
            clf = svm.SVC(probability=True, kernel='poly', degree=1)
        if clf_type == 'polysvc4':
            clf = svm.SVC(probability=True, kernel='poly', degree=4)
        if clf_type == 'polysvc6':
            clf = svm.SVC(probability=True, kernel='poly', degree=6)

        clf = clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probas = clf.predict_proba(X_test)
            
        cv_target = np.concatenate((cv_target, y_test), axis=0) # concatenar los resultados
        cv_prediction = np.concatenate((cv_prediction, preds), axis=0)
        cv_probas = np.concatenate((cv_probas, probas[:,1]), axis=0)
            

    preds_thr = np.zeros(len(cv_target))
    thr_final = get_optimal_thr_diagonal_cm(cv_probas, cv_target, 0.01)
    preds_thr[np.where(cv_probas>thr_final)[0]] = 1
    cm = confusion_matrix(cv_target, preds_thr).astype(float)
    cm[0,:] = cm[0,:]/float(sum(cm[0,:])) # obtener matricz de confusion normalizada
    cm[1,:] = cm[1,:]/float(sum(cm[1,:]))
            
        
    fpr, tpr, thresholds = roc_curve(cv_target,  cv_probas) # obtener la curva ROC

    return auc(fpr,tpr), clf # area de la curva ROC


def eval_set(clf):
    ########### EVALUAR EN DATOS DE PROPOFOL #####################  

    # cargar matrices de correlacion
    with open('datasets/propofol_data.pickle', "rb") as input_file:
       propofol_data = pickle.load(input_file)
       
    set1 = unfold_data(propofol_data['W'])  
    set2 = unfold_data(propofol_data['LOC'])

    target1 = np.zeros(set1.shape[0])
    target2 = np.ones(set2.shape[0])
    	
    target = np.concatenate((target1, target2), axis=0)
    data = np.concatenate((set1,set2), axis=0)
    probas = clf.predict_proba(data)[:,1]

    preds_thr = np.zeros(len(target))
    thr_final = get_optimal_thr_diagonal_cm(probas, target, 0.01)
    preds_thr[np.where(probas>thr_final)[0]] = 1
    cm_propofol = confusion_matrix(target, preds_thr).astype(float)
    cm_propofol[0,:] = cm_propofol[0,:]/float(sum(cm_propofol[0,:])) # obtener matricz de confusion normalizada
    cm_propofol[1,:] = cm_propofol[1,:]/float(sum(cm_propofol[1,:]))
            
        
    fpr_propofol, tpr_propofol, thresholds_propofol = roc_curve(target,  probas) # obtener la curva ROC

    return auc(fpr_propofol,tpr_propofol) # area de la curva ROC