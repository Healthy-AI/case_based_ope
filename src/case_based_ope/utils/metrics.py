import numpy as np
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

def ece(y, y_probas, n_bins=10):
    '''
    Parameters
    ----------
    y : NumPy array of shape (n_samples, n_classes) or (n_samples,)
        True labels. Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).
    y_probas : NumPy array of shape (n_samples, n_classes) or (n_samples,)
        Predicted probabilities.

    Returns
    -------
    ece : float
        ECE score.
    '''    
    if y_probas.ndim == 2:
        jj = y_probas.argmax(axis=-1)
        if y.ndim == 2:
            y = np.take_along_axis(y, np.expand_dims(jj, axis=-1), axis=-1).squeeze(axis=-1)
        elif not np.array_equal(y, y.astype(bool)):
            assert set(y) == set(jj), 'The labels must range from 0 to n_classes-1.' 
            y = np.array([yi == j for yi, j in zip(y, jj)], dtype=bool)
        y_probas = np.take_along_axis(y_probas, np.expand_dims(jj, axis=-1), axis=-1).squeeze(axis=-1)
    else:
        assert y.ndim == 1 and np.array_equal(y, y.astype(bool))
        
    bins = np.linspace(0, 1, n_bins+1)
    bin_indices = np.digitize(y_probas, bins, right=False)  # Assume no probabilities = 1

    ece_ = 0
    for i in range(1, n_bins+1):
        bin_mask = bin_indices == i
        if bin_mask.sum() == 0:
            continue
        y_probas_bin = y_probas[bin_mask]
        conf = np.mean(y_probas_bin)
        y_bin = y[bin_mask]     
        acc = np.mean(y_bin)
        ece_ += len(y_bin)*np.abs(acc-conf)

    return ece_/len(y_probas)

def sce(y_one_hot, y_probas, n_bins=10):
    sce_ = 0
    for yp, y in zip(y_probas.T, y_one_hot.T):
        sce_ += ece(y, yp, n_bins)
    return sce_ / y_probas.shape[1]

def auc_multiclass(y, y_probas):
    return roc_auc_score(y, y_probas, multi_class='ovr')

def cv_scorer(pipeline, X, y):
    out = {}

    net = pipeline[-1]
    default = net.n_prediction_prototypes
    
    for n_prediction_prototypes in range(1, 5+1):
        net.n_prediction_prototypes = n_prediction_prototypes
        y_probas = pipeline.predict_proba(X)
        key = 'neg_log_loss_' + str(n_prediction_prototypes)
        out[key] = -log_loss(y, y_probas)

    net.n_prediction_prototypes = default
        
    return out
