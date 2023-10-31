from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc,f1_score
import numpy as np
from joblib import Parallel, delayed

def find_best_threshold_par(y_true, y_prob, thresholds, n_jobs = -1):

    def return_fscore(threshold):
        y_pred = (y_prob >= threshold).astype(int)
        return f1_score(y_true, y_pred, average='binary')
    
    best = np.argmax(Parallel(n_jobs = n_jobs)(delayed(return_fscore)(threshold) for threshold in thresholds))
    return best

def scorer(y_true, y_prob,n_jobs = -1):
    """
    Scorer given the y_true and y_prob
    """

    if len(y_true) != len(y_prob):
        raise IndexError('len(y_true) !=  len(y_prob)')
    

    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label = 1)
    metric_auc = auc(fpr,tpr)
    
    best = find_best_threshold_par(y_true, y_prob, thresholds, n_jobs = -1)
    
    threshold =  thresholds[best]
    
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division= 0.0)
    acc =  accuracy_score(y_true, y_pred)

    return threshold, thresholds, {'Accuracy': acc, 'Recall': recall, 'Precision':precision, 'Fscore':fscore, 'AUC':metric_auc}
