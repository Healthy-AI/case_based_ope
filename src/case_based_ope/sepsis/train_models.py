import torch
from case_based_ope.utils.misc import get_args
from case_based_ope.utils.misc import load_config
from case_based_ope.utils.misc import save_data
from case_based_ope.models.training import *
from case_based_ope.sepsis.data import Data
from amhelpers.config_parsing import get_net_params

def main(config):
    ## Load data

    train_data, valid_data, _ = Data.get_train_valid_test_data(
        config.data.path,
        config.ai_clinician.path,
        config.data.valid_size,
        config.data.test_size,
        config.data.use_aic_split,
        config.data.seed
    )

    Xg_train, y_train, _, _ = train_data
    X_train = Xg_train.drop(columns=Data.C_GROUP)

    Xg_valid, y_valid, _, _ = valid_data
    X_valid = Xg_valid.drop(columns=Data.C_GROUP)

    input_size = X_train.shape[1]
    output_size = len(set(y_train))

    preprocessor = Data.get_preprocessor()
    transformers = [('preprocessor', preprocessor)]

    ## Train models

    if torch.cuda.is_available():
        config.default.device = 'gpu'

    # Fit LR
    print('Fitting LR'); sys.stdout.flush()
    lr = fit_cal_lr(
        estimator_kwargs=config.lr.estimator_kwargs,
        transformers=transformers,
        search_kwargs=config.lr.search_kwargs,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid
    )

    # Fit RF
    print('Fitting RF'); sys.stdout.flush()
    rf = fit_cal_rf(
        estimator_kwargs=config.rf.estimator_kwargs,
        transformers=transformers,
        search_kwargs=config.rf.search_kwargs,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid
    )

    # Fit MLP
    print('Fitting MLP'); sys.stdout.flush()
    mlp_estimator_kwargs = get_net_params(config.default, config.mlp.estimator_kwargs)
    mlp_estimator_kwargs.update({'module__encoder__input_size': input_size, 'module__output_size': output_size})
    mlp = fit_cal_mlp(
        estimator_kwargs=mlp_estimator_kwargs,
        transformers=transformers,
        search_kwargs=config.mlp.search_kwargs,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid
    )

    # Fit RNN
    print('Fitting RNN'); sys.stdout.flush()
    rnn_estimator_kwargs = get_net_params(config.default, config.rnn.estimator_kwargs)
    rnn_estimator_kwargs.update({'module__encoder__input_size': input_size, 'module__output_size': output_size})
    rnn = fit_cal_rnn(
        estimator_kwargs=rnn_estimator_kwargs,
        transformers=transformers,
        search_kwargs=config.rnn.search_kwargs,
        X_train=Xg_train,
        y_train=y_train,
        X_valid=Xg_valid,
        y_valid=y_valid
    )

    # Fit ProNet
    print('Fitting ProNet'); sys.stdout.flush()
    pronet_estimator_kwargs = get_net_params(config.default, config.pronet.estimator_kwargs)
    pronet_estimator_kwargs.update({'module__encoder__input_size': input_size, 'module__output_size': output_size})
    pronet_models = fit_cal_pronet(
        estimator_kwargs=pronet_estimator_kwargs,
        transformers=transformers,
        param_grid=config.pronet.param_grid,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        n_iterations=config.n_inits
    )

    # Fit ProSeNet
    print('Fitting ProSeNet'); sys.stdout.flush()
    prosenet_estimator_kwargs = get_net_params(config.default, config.prosenet.estimator_kwargs)
    prosenet_estimator_kwargs.update({'module__encoder__input_size': input_size, 'module__output_size': output_size})
    prosenet_models = fit_cal_prosenet(
        estimator_kwargs=prosenet_estimator_kwargs,
        transformers=transformers,
        param_grid=config.prosenet.param_grid,
        X_train=Xg_train,
        y_train=y_train,
        X_valid=Xg_valid,
        y_valid=y_valid,
        n_iterations=config.n_inits
    )

    ## Save models

    models_dir = config.results.path + 'models/'

    save_data(lr, models_dir, 'lr_cal')
    save_data(rf, models_dir, 'rf_cal')
    save_data(mlp, models_dir, 'mlp_cal')
    save_data(rnn, models_dir, 'rnn_cal')

    for pronet_model, n_training_prototypes, n_prediction_prototypes, iteration in pronet_models:
        filename = 'pronet_cal_{}_{}_{}'.format(n_training_prototypes, n_prediction_prototypes, iteration)
        save_data(pronet_model, models_dir, filename)
    
    for prosenet_model, n_training_prototypes, n_prediction_prototypes, iteration in prosenet_models:
        filename = 'prosenet_cal_{}_{}_{}'.format(n_training_prototypes, n_prediction_prototypes, iteration)
        save_data(prosenet_model, models_dir, filename)

if __name__ == '__main__':
    args = get_args()
    config_path = args.config_path
    config = load_config(config_path)

    if torch.cuda.device_count() > 1:
        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client
        cluster = LocalCUDACluster()
        client = Client(cluster)
        main(config)
        for worker in cluster.workers.values():
            process = worker.process.process
            if process.is_alive(): process.terminate()
        client.shutdown()
    else:
        main(config)
