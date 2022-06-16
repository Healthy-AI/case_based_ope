import sys
import torch
import sklearn
import numpy as np
import pandas as pd
import case_based_ope.sepsis_sim.gumbel_max_scm.cf.counterfactual as cf
from case_based_ope.sepsis_sim.utils import Policy
from case_based_ope.sepsis_sim.utils import soften_policy
from case_based_ope.sepsis_sim.utils import compute_true_reward
from case_based_ope.sepsis_sim.utils import sample_data_from_policy
from case_based_ope.sepsis_sim.utils import learn_true_behavior_policy
from case_based_ope.sepsis_sim.utils import learn_target_policy
from case_based_ope.sepsis_sim.gumbel_max_scm.sepsisSimDiabetes.Action import Action
from case_based_ope.policy_evaluation.importance_sampling import evaluate_multiple_policies
from case_based_ope.utils.misc import save_data
from sklearn.compose import ColumnTransformer
from case_based_ope.models.training import *
from amhelpers.config_parsing import get_net_params
from case_based_ope.utils.misc import load_config
from case_based_ope.utils.misc import get_args

def main(config):
    ## Learn the behavior policy

    mu_true_deterministic, _ = learn_true_behavior_policy(config.discount_pol, config.data.mdp_params_path)
    mu_true_soft = soften_policy(mu_true_deterministic, config.mu_epsilon)

    ## Main loop

    if torch.cuda.is_available():
        config.default.device = 'gpu'

    data = {
        'time': [],
        'model': [],
        'gt': [],
        'v_diff': [],
        'w_ratio': []
    }

    weights = {}

    for n_steps in range(config.min_n_steps, config.max_n_steps+1, 5):
        print('Trajectory length:', n_steps); sys.stdout.flush()

        weights[n_steps] = []

        #Estimate the value of the behavior policy
        #mu_true_reward = compute_true_reward(
        #    mu_true_soft,
        #    config.n_true_samples, 
        #    n_steps,
        #    policy_idx_type='full',
        #    p_diabetes=config.p_diabetes,
        #    discount=config.discount,
        #    n_bootstrap=None
        #)

        for iteration in range(1, config.n_iterations+1):
            print('Iteration', iteration, 'of', config.n_iterations); sys.stdout.flush()

            ## Sample data from the behavior policy

            # Training data
            train_data, emp_tx, emp_r, Xg_train, _, y_train, _  = sample_data_from_policy(
                mu_true_soft,
                config.n_train_samples,
                n_steps,
                config.p_diabetes,
                n_sa_pairs=config.n_train_sa_pairs,
                policy_idx_type='full',
                output_state_idx_type=config.state_idx_type
            )
            X_train = Xg_train.iloc[:, :-1].to_numpy()

            # Validation data
            _, _, _, Xg_valid, _, y_valid, _  = sample_data_from_policy(
                mu_true_soft,
                config.n_valid_samples,
                n_steps,
                config.p_diabetes,
                n_sa_pairs=config.n_valid_sa_pairs,
                policy_idx_type='full',
                output_state_idx_type=config.state_idx_type
            )
            X_valid = Xg_valid.iloc[:, :-1].to_numpy()

            # Test data
            test_data, _, _, Xg_test, states_test, y_test, rewards_test = sample_data_from_policy(
                mu_true_soft,
                config.n_test_samples,
                n_steps,
                config.p_diabetes,
                n_sa_pairs=config.n_test_sa_pairs,
                policy_idx_type='full',
                output_state_idx_type=config.state_idx_type
            )
            X_test = Xg_test.iloc[:, :-1].to_numpy()

            ## Estimate the behavior policy

            scale_tf = sklearn.pipeline.Pipeline(steps=[('scale', sklearn.preprocessing.StandardScaler())])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('scale', scale_tf, Xg_train.columns[:-1]),  # The last column identifies the sequence
                ],
                remainder='passthrough'
            )
            transformers = [('preprocessor', preprocessor)]

            input_size = X_train.shape[1]
            output_size = Action.NUM_ACTIONS_TOTAL

            # MLP classifier
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

            # ProNet classifier
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
                n_iterations=1
            )
            
            ## Learn a target policy from samples

            pi_deterministic, _, _ = learn_target_policy(train_data, emp_tx, emp_r, config.discount_pol, config.data.mdp_params_path)
            pi_soft = soften_policy(pi_deterministic, config.pi_epsilon)

            pi_true_reward = compute_true_reward(
                pi_soft[:-2, :],  # We should ignore absorbing states (last two rows)
                config.n_true_samples,
                n_steps,
                policy_idx_type=config.state_idx_type,
                p_diabetes=config.p_diabetes,
                discount=config.discount,
                n_bootstrap=None
            )

            ## Evaluate the target policy using WIS

            target_policies = {'pi_soft': Policy(pi_soft)}
            
            mu_models = {'mu_true': Policy(mu_true_soft), 'mlp': mlp}
            for pronet_model, n_training_prototypes, n_prediction_prototypes, _ in pronet_models:
                model_str = 'pronet_'+ '_'.join([str(i) for i in [n_training_prototypes, n_prediction_prototypes]])
                mu_models[model_str] = pronet_model

            policy_inputs = dict.fromkeys(list(target_policies.keys()) + list(mu_models.keys()))
            for policy in policy_inputs.keys():
                if policy.startswith('pi') or policy.startswith('mu'):
                    policy_inputs[policy] = states_test
                else:
                    policy_inputs[policy] = X_test

            sequence_indices = [group[1].index.tolist() for group in Xg_test.groupby(by=Xg_test.columns[-1])]

            _, wis_estimates, weights_, _ = evaluate_multiple_policies(
                target_policies,
                mu_models,
                policy_inputs,
                y_test,
                rewards_test,
                config.discount,
                sequence_indices,
                n_bootstrap_samples=config.n_bootstrap
            )

            weights[n_steps] += [weights_]

            gt = np.mean(pi_true_reward)  # "ground truth"

            for mu, wis_estimates_ in wis_estimates['pi_soft'].items():  # Loop over models
                for i, estimate in enumerate(wis_estimates_):  # Loop over bootstraps
                    w_ratios = np.divide(weights_['pi_soft'][mu][i], weights_['pi_soft']['mu_true'][i])
                    n = len(w_ratios)
                    data['time'] += n*[n_steps]
                    data['model'] += n*[mu]
                    data['gt'] += n*[gt]
                    data['v_diff'] += n*[np.abs(gt-estimate)]
                    for w_ratio in w_ratios:
                        data['w_ratio'] += [w_ratio]

            #Esimate the target policy using the true behavior policy
            #use_bootstrap = config.n_bootstrap > 1
            #wis_estimates_ = cf.eval_wis(
            #    test_data,
            #    discount=config.discount,
            #    bootstrap=use_bootstrap,
            #    n_bootstrap=config.n_bootstrap,
            #    obs_policy=mu_true_soft,
            #    new_policy=pi_soft
            #)[0]
            #for estimate in wis_estimates_:  # Loop over bootstraps
            #    data['time'] += [n_steps]
            #    data['model'] += ['mu_true']
            #    data['v_diff'] += [np.abs(gt-estimate)]
            #    data['w_ratio'] += [None]

    ## Save data

    save_data(weights, config.results.path, 'weights.pickle')
    data_df = pd.DataFrame(data)
    data_df.to_csv(config.results.path + 'wis_data.csv', index=False)

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
