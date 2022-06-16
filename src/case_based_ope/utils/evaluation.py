import skorch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from case_based_ope.utils.metrics import ece
from case_based_ope.utils.metrics import sce
from case_based_ope.utils.metrics import auc_multiclass

def plot_metric_against_epoch(net, metric='tot_loss'):
    assert isinstance(net, skorch.NeuralNet)
    assert 'train_' + metric in net.history[-1].keys() 
    data_train = [(stats['epoch'], stats['train_'+metric]) for stats in net.history]
    try:
        data_valid = [(stats['epoch'], stats['valid_'+metric]) for stats in net.history]
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(*zip(*data_train))
        ax1.set_title('train')
        ax2.plot(*zip(*data_valid))
        ax2.set_title('valid')
    except KeyError:
        fig, ax = plt.subplots()
        ax.plot(*zip(*data_train))
        ax.set_title('train')
    return fig

def get_scores(models, inputs, targets, n_bootstraps):
    scores = {}
    for model_name, model in models.items():
        probas = model.predict_proba(inputs[model_name])
        scores[model_name] = evaluate(targets, probas, n_bootstraps)
    return scores

def evaluate(y, y_probas, n_bootstraps):
    y_preds = y_probas.argmax(axis=1)

    n_classes = y_probas.shape[1]

    y = list(map(int, y))
    y_one_hot = np.eye(n_classes)[y]

    if n_classes == 2:
        score_functions = {
            'accuracy': accuracy_score,
            'ece': ece,
            'auc': roc_auc_score
        }
        y_probas = y_probas[:, 1]
    else:
        score_functions = {
            'accuracy': accuracy_score,
            'sce': sce,
            'auc': auc_multiclass
        }
    
    scores = {}

    for metric, score_function in score_functions.items():
        if metric == 'accuracy':
            yt, yp = y, y_preds
        elif metric == 'log_loss' or metric == 'auc':
            yt, yp = y, y_probas
        elif n_classes == 2 and metric == 'ece':
            yt, yp = y, y_probas
        elif n_classes > 2 and (metric == 'ece' or metric == 'sce'):
            yt, yp = y_one_hot, y_probas
        
        if n_bootstraps > 0:
            scores[metric] = _score_stat_ci(yt, [yp], score_function, n_bootstraps=n_bootstraps)
        else:
            scores[metric] = score_function(yt, yp)

    return scores

def _score_stat_ci(
    y_true,
    y_preds,
    score_fun,
    stat_fun=np.mean,
    n_bootstraps=2000,
    confidence_level=0.95,
    seed=None,
    reject_one_class_samples=True,
):
    '''
    Compute confidence interval for given statistic of a score function based on labels and 
    predictions using bootstrapping.
    Reference: https://github.com/mateuszbuda/ml-stat-util/blob/master/stat_util.py.
    '''
    y_true = np.array(y_true)
    y_preds = np.atleast_2d(y_preds)
    assert all(len(y_true) == len(y) for y in y_preds)

    np.random.seed(seed)
    scores = []
    for _ in range(n_bootstraps):
        readers = np.random.randint(0, len(y_preds), len(y_preds))
        indices = np.random.randint(0, len(y_true), len(y_true))
        if reject_one_class_samples and len(np.unique(y_true[indices])) < 2:
            continue
        reader_scores = []
        for r in readers:
            reader_scores.append(score_fun(y_true[indices], y_preds[r][indices]))
        scores.append(stat_fun(reader_scores))

    mean_score = np.mean(scores)
    sorted_scores = np.array(sorted(scores))
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = sorted_scores[int(round(alpha * len(sorted_scores)))]
    ci_upper = sorted_scores[int(round((1.0 - alpha) * len(sorted_scores)))]
    
    return mean_score, ci_lower, ci_upper
