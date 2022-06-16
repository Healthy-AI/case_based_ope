import numpy as np

def evaluate_multiple_policies(
    target_policies,
    behavior_policies,
    policy_inputs,
    actions,
    rewards,
    discount,
    sequence_indices,
    n_bootstrap_samples,
    verb=0,
    pi_until=None,
    weight_threshold=None
):
    is_estimates = {}
    wis_estimates = {}
    weights = {}
    ratios = {}

    if pi_until is None:
        pi_until = max([len(ii) for ii in sequence_indices])

    for pi in target_policies.keys():
        is_estimates[pi] = {mu: [] for mu in behavior_policies.keys()}
        wis_estimates[pi] = {mu: [] for mu in behavior_policies.keys()}
        weights[pi] = {mu: [] for mu in behavior_policies.keys()}
        ratios[pi] = {mu: [] for mu in behavior_policies.keys()}

    n_sequences = len(sequence_indices)

    if n_bootstrap_samples == 1:
        indices = np.arange(n_sequences).reshape(1, -1)
    else:
        indices = np.random.randint(low=0, high=n_sequences, size=(n_bootstrap_samples, n_sequences))

    for pi, pi_model in target_policies.items():
        probas_pi = pi_model.predict_proba(policy_inputs[pi])

        for mu, mu_model in behavior_policies.items():
            if verb > 0: print(pi, mu)

            probas_mu = mu_model.predict_proba(policy_inputs[mu])

            is_zero = probas_mu == 0
            if np.sum(is_zero) > 0:
                probas_mu[is_zero] += 1e-6

            probas_pi_ = []
            probas_mu_ = []
            rewards_ = []

            for ii_sequence in sequence_indices:
                assert isinstance(ii_sequence, list)
                actions_sequence = actions[ii_sequence]

                probas_pi_ += [
                    [probas_pi[i, a] if t < pi_until else probas_mu[i, a] for t, (i, a) in enumerate(zip(ii_sequence, actions_sequence))]
                ]
                probas_mu_ += [[probas_mu[i, a] for i, a in zip(ii_sequence, actions_sequence)]]
                rewards_sequence = rewards[ii_sequence]
                rewards_ += [rewards_sequence]

            for ii in indices:
                is_estimate, wis_estimate, weights_, ratios_ = perform_importance_sampling(
                    [probas_pi_[i] for i in ii],
                    [probas_mu_[i] for i in ii],
                    [rewards_[i] for i in ii],
                    discount,
                    weight_threshold
                )
                is_estimates[pi][mu] += [is_estimate]
                wis_estimates[pi][mu] += [wis_estimate]
                weights[pi][mu] += [weights_]
                ratios[pi][mu] += [ratios_]

    return is_estimates, wis_estimates, weights, ratios

def perform_importance_sampling(probas_pi, probas_mu, rewards, gamma=1, weight_threshold=None):
    weights = []
    ratios = []

    estimate = 0
    for probas_pi_, probas_mu_, rewards_ in zip(probas_pi, probas_mu, rewards):
        ratios_ = np.divide(probas_pi_, probas_mu_)
        w = np.prod(ratios_)
        if weight_threshold is not None and w > weight_threshold:
            continue
        ratios += [ratios_]
        weights += [w]
        G = np.sum([(gamma**t)*r for t, r in enumerate(rewards_)])
        estimate += w*G

    n_samples = len(weights)
    is_estimate = estimate / n_samples

    wis_estimate = estimate / np.sum(weights)

    return is_estimate, wis_estimate, weights, ratios
