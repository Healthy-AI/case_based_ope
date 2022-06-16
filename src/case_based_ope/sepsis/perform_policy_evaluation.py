from case_based_ope.utils.misc import get_args
from case_based_ope.utils.misc import load_config
from case_based_ope.utils.postprocessing import get_model_paths_from_directory
from case_based_ope.utils.postprocessing import load_models
from case_based_ope.sepsis.data import Data
from case_based_ope.utils.misc import save_data
from os.path import join
from case_based_ope.sepsis.target_policies import AICPolicy
from case_based_ope.sepsis.target_policies import ZeroDrugPolicy
from case_based_ope.sepsis.target_policies import RandomPolicy
from case_based_ope.policy_evaluation.importance_sampling import evaluate_multiple_policies

def main(config):
    ## Load data

    _, _, test_data = Data.get_train_valid_test_data(
        config.data.path,
        config.ai_clinician.path,
        config.data.valid_size,
        config.data.test_size,
        config.data.use_aic_split,
        config.data.seed
    )

    Xg_test, y_test, states_test, rewards_test = test_data
    Xg_test.reset_index(drop=True, inplace=True)  # Important for extracting sequence indices below
    X_test = Xg_test.drop(columns=Data.C_GROUP)

    ## Load models of the behavior policy

    model_dir = config.results.path + 'models/'
    model_paths = get_model_paths_from_directory(model_dir)
    mu_models = load_models(model_paths)

    policy_inputs = {}
    for model_name in mu_models.keys():
        if model_name in ['lr', 'rf', 'mlp'] or model_name.startswith('pronet'):
            policy_inputs[model_name] = X_test
        elif model_name in ['rnn'] or model_name.startswith('prosenet'):
            policy_inputs[model_name] = Xg_test
    
    ## Create target policies

    aic_policy_path = join(config.ai_clinician.path, 'target_policy.csv')

    target_policies = {
        'aic': AICPolicy(aic_policy_path),
        'zero_drug': ZeroDrugPolicy(),
        'random': RandomPolicy()
    }

    for pi in target_policies.keys():
        policy_inputs[pi] = states_test

    ## Perform policy evaluation

    sequence_indices_test = [group[1].index.tolist() for group in Xg_test.groupby(by=Data.C_GROUP)]

    is_estimates, wis_estimates, weights, _ = evaluate_multiple_policies(
        target_policies,
        mu_models,
        policy_inputs,
        y_test,
        rewards_test,
        config.discount,
        sequence_indices_test,
        n_bootstrap_samples=config.n_is_bootstraps,
        pi_until=config.pi_until,
        weight_threshold=config.weight_threshold
    )
    
    ## Save data

    save_data(is_estimates, config.results.path, 'is_estimates')
    save_data(wis_estimates, config.results.path, 'wis_estimates')
    save_data(weights, config.results.path, 'weights')

if __name__ == '__main__':
    args = get_args()
    config_path = args.config_path
    config = load_config(config_path)
    main(config)
