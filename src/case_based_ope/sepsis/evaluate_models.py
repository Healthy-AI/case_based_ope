from case_based_ope.utils.misc import get_args
from case_based_ope.utils.misc import load_config
from case_based_ope.utils.postprocessing import get_model_paths_from_directory
from case_based_ope.utils.postprocessing import load_models
from case_based_ope.sepsis.data import Data
from case_based_ope.utils.misc import save_data
from case_based_ope.utils.evaluation import get_scores
from case_based_ope.models.posthoc_clustering import PostHocClustering
from sklearn.calibration import CalibratedClassifierCV

def main(config):
    train_data, valid_data, test_data = Data.get_train_valid_test_data(
        config.data.path,
        config.ai_clinician.path,
        config.data.valid_size,
        config.data.test_size,
        config.data.use_aic_split,
        config.data.seed
    )

    Xg_test, y_test, _, _ = test_data
    X_test = Xg_test.drop(columns=Data.C_GROUP)

    model_dir = config.results.path + 'models/'
    model_paths = get_model_paths_from_directory(model_dir)
    models = load_models(model_paths)

    model_inputs = {}
    for model_name in models.keys():
        if model_name in ['lr', 'rf', 'mlp'] or model_name.startswith('pronet'):
            model_inputs[model_name] = X_test
        elif model_name in ['rnn'] or model_name.startswith('prosenet'):
            model_inputs[model_name] = Xg_test

    if 'rnn' in models:
        Xg_train, y_train, _, _ = train_data
        Xg_valid, y_valid, _, _ = valid_data

        phc = PostHocClustering(
            models['rnn'],
            n_clusters=10
        )
        phc.fit(Xg_train, y_train)
        phc_cal = CalibratedClassifierCV(base_estimator=phc, method='sigmoid', cv='prefit').fit(Xg_valid, y_valid)
        models.update({'posthoc': phc_cal})
        model_inputs.update({'posthoc': Xg_test})

    scores = get_scores(models, model_inputs, y_test, config.n_scoring_bootstraps)
    
    save_data(scores, config.results.path, 'scores')

if __name__ == '__main__':
    args = get_args()
    config_path = args.config_path
    config = load_config(config_path)
    main(config)
