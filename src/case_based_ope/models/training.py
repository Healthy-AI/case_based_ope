import sys
import time
import itertools
import copy
import torch
import pandas as pd
from sklearn.model_selection import GridSearchCV
from case_based_ope.utils.metrics import cv_scorer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from case_based_ope.models.prototypes import PrototypeClassifier

def _make_pipeline(transformers, estimator):
    transformers_ = copy.deepcopy(transformers)
    steps = transformers_ + [('estimator', estimator)]
    return Pipeline(steps=steps)

def _get_model(Estimator, estimator_kwargs, transformers):
    estimator = Estimator(**estimator_kwargs)
    if transformers:
        return _make_pipeline(transformers, estimator)
    else:
        return estimator

def _fit_cal_model(
    estimator,
    search_kwargs,
    X_train,
    y_train,
    X_valid,
    y_valid,
):
    if search_kwargs is not None:
        search = GridSearchCV(estimator, **search_kwargs)
        search.fit(X_train, y_train)
        assert hasattr(search, 'best_estimator_')
        calibrated = CalibratedClassifierCV(base_estimator=search, method='sigmoid', cv='prefit')
    else:
        estimator.fit(X_train, y_train)
        calibrated = CalibratedClassifierCV(base_estimator=estimator, method='sigmoid', cv='prefit')
    return calibrated.fit(X_valid, y_valid)

def _train_best_prototype_model(
    estimator,
    cv_results,
    n_training_prototypes,
    n_prediction_prototypes,
    X_train,
    y_train,
    X_valid,
    y_valid,
    iteration
):
    best_params = cv_results[
        cv_results['param_estimator__module__n_prototypes'] == n_training_prototypes
    ].sort_values('rank_test_neg_log_loss_%d' % n_prediction_prototypes)['params'].iloc[0]
    estimator.set_params(**best_params)
    estimator.fit(X_train, y_train)
    estimator[-1].n_prediction_prototypes = n_prediction_prototypes
    estimator_cal = CalibratedClassifierCV(base_estimator=estimator, method='sigmoid', cv='prefit')
    estimator_cal.fit(X_valid, y_valid)
    return (estimator_cal, n_training_prototypes, n_prediction_prototypes, iteration)

def _fit_cal_prototype_model(
    estimator_getter,
    estimator_kwargs,
    transformers,
    param_grid,
    X_train,
    y_train,
    X_valid,
    y_valid,
    n_iterations
):
    search = GridSearchCV(
        estimator_getter(estimator_kwargs=estimator_kwargs, transformers=transformers), 
        param_grid,
        scoring=cv_scorer, 
        cv=3,
        refit=False,
        error_score='raise'
    )

    n_prototypes = param_grid['estimator__module__n_prototypes']

    print('Searching for optimal hyperparameters.'); sys.stdout.flush()
    start_time = time.time()
    if torch.cuda.device_count() > 1:
        from joblib import parallel_backend
        with parallel_backend('dask'):
            search.fit(X_train, y_train)
    else:
        search.fit(X_train, y_train)
    print('--- %.2f min ---' % ((time.time() - start_time) / 60)); sys.stdout.flush()

    cv_results = pd.DataFrame(search.cv_results_)

    print('Training optimal models.'); sys.stdout.flush()
    start_time = time.time()
    iterator = itertools.product(n_prototypes, range(1, 5+1), range(1, n_iterations+1))
    if torch.cuda.device_count() > 1:
        from joblib import Parallel, delayed
        with parallel_backend('dask'):
            optimal_models = Parallel()(
                delayed(_train_best_prototype_model)(
                    estimator_getter(estimator_kwargs=estimator_kwargs, transformers=transformers),
                    cv_results,
                    n_training_prototypes,
                    n_prediction_prototypes,
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    i
                )
                for n_training_prototypes, n_prediction_prototypes, i in iterator
            )
    else:
        optimal_models = []
        for n_training_prototypes, n_prediction_prototypes, i in iterator:
            optimal_models.append(
                _train_best_prototype_model(
                    estimator_getter(estimator_kwargs=estimator_kwargs, transformers=transformers),
                    cv_results,
                    n_training_prototypes,
                    n_prediction_prototypes,
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    i
                )
            )
    print('--- %.2f min ---' % ((time.time() - start_time) / 60)); sys.stdout.flush()

    return optimal_models

def get_lr(**kwargs):
    from sklearn.linear_model import LogisticRegression
    return _get_model(LogisticRegression, **kwargs)

def get_rf(**kwargs):
    from sklearn.ensemble import RandomForestClassifier
    return _get_model(RandomForestClassifier, **kwargs)

def get_mlp(**kwargs):
    return _get_model(PrototypeClassifier, **kwargs)

def get_rnn(**kwargs):
    return _get_model(PrototypeClassifier, **kwargs)

def get_pronet(**kwargs):
    return _get_model(PrototypeClassifier, **kwargs)

def get_prosenet(**kwargs):
    return _get_model(PrototypeClassifier, **kwargs)

def fit_cal_lr(estimator_kwargs, transformers, **kwargs):
    lr = get_lr(estimator_kwargs=estimator_kwargs, transformers=transformers)
    return _fit_cal_model(lr, **kwargs)

def fit_cal_rf(estimator_kwargs, transformers, **kwargs):
    rf = get_rf(estimator_kwargs=estimator_kwargs, transformers=transformers)
    return _fit_cal_model(rf, **kwargs)

def fit_cal_mlp(estimator_kwargs, transformers, **kwargs):
    mlp = get_mlp(estimator_kwargs=estimator_kwargs, transformers=transformers)
    return _fit_cal_model(mlp, **kwargs)

def fit_cal_rnn(estimator_kwargs, transformers, **kwargs):
    rnn = get_rnn(estimator_kwargs=estimator_kwargs, transformers=transformers)
    return _fit_cal_model(rnn, **kwargs)

def fit_cal_pronet(**kwargs):
    return _fit_cal_prototype_model(get_pronet, **kwargs)

def fit_cal_prosenet(**kwargs):
    return _fit_cal_prototype_model(get_prosenet, **kwargs)
