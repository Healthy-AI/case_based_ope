import pickle
import numpy as np
import pandas as pd
import case_based_ope.sepsis_sim.gumbel_max_scm.cf.counterfactual as cf
import case_based_ope.sepsis_sim.gumbel_max_scm.cf.utils as utils
from case_based_ope.sepsis_sim.gumbel_max_scm.sepsisSimDiabetes.State import State
from case_based_ope.sepsis_sim.gumbel_max_scm.sepsisSimDiabetes.Action import Action
from case_based_ope.sepsis_sim.gumbel_max_scm.sepsisSimDiabetes.DataGenerator import DataGenerator
from scipy.linalg import block_diag

# ----------------------------------------------------------
# -- General -----------------------------------------------
# ----------------------------------------------------------

class Policy():
    def __init__(self, policy):
        self.policy = policy
        
    def predict_proba(self, states):
        probas = []
        for s in states:
            probas.append(self.policy[int(s), :])
        return np.array(probas)

def soften_policy(policy, epsilon):
    n_actions = policy.shape[1]
    soft_policy = np.copy(policy)
    soft_policy[soft_policy == 1] = 1 - epsilon
    soft_policy[soft_policy == 0] = epsilon / (n_actions - 1)
    return soft_policy

def compute_true_reward(policy, n_samples, n_steps, policy_idx_type, p_diabetes, discount, n_bootstrap=None):
    '''
    Note: Remove any absorbing states before passing a policy to this function.
    '''
    states, actions, _, rewards, diab, _, _ = DataGenerator().simulate(
        n_samples,
        n_steps,
        policy=policy,
        policy_idx_type=policy_idx_type, 
        p_diabetes=p_diabetes,
        output_state_idx_type=policy_idx_type,
        use_tqdm=False
    )
    
    obs_samps = utils.format_dgen_samps(states, actions, rewards, diab, n_steps, n_samples)
    
    use_bootstrap = (n_bootstrap is not None and n_bootstrap > 1)
    true_reward = cf.eval_on_policy(obs_samps, discount=discount, bootstrap=use_bootstrap, n_bootstrap=n_bootstrap)

    return true_reward

# ----------------------------------------------------------
# -- Data --------------------------------------------------
# ----------------------------------------------------------

def sample_data_from_policy(policy, n_samples, n_steps, p_diabetes, n_sa_pairs=None, policy_idx_type='full', output_state_idx_type='obs'):
    dgen = DataGenerator()
    
    if n_sa_pairs is None:
        states, actions, _, rewards, diab, emp_tx, emp_r = dgen.simulate(
            n_samples,
            n_steps,
            policy=policy,
            policy_idx_type=policy_idx_type,
            p_diabetes=p_diabetes,
            output_state_idx_type=output_state_idx_type,
            use_tqdm=False
        )
    else:
        states = []
        actions = []
        rewards = []
        diab = []
        emp_tx = []
        emp_r = []
        tot_length = 0

        n_samples_ = int(0.1 * (n_sa_pairs / n_steps))

        while tot_length < n_sa_pairs:
            states_, actions_, lengths_, rewards_, diab_, emp_tx_, emp_r_ = dgen.simulate(
                n_samples_,
                n_steps,
                policy=policy,
                policy_idx_type=policy_idx_type,
                p_diabetes=p_diabetes,
                output_state_idx_type=output_state_idx_type,
                use_tqdm=False
            )
            states += [states_]
            actions += [actions_]
            rewards += [rewards_]
            diab += [diab_]
            emp_tx += [emp_tx_]
            emp_r += [emp_r_]
            tot_length += np.sum(lengths_)

        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        diab = np.concatenate(diab, axis=0)
        emp_tx = np.sum(emp_tx, axis=0)
        emp_r = np.sum(emp_r, axis=0)

    compact_data = utils.format_dgen_samps(states, actions, rewards, diab, n_steps, states.shape[0])

    Xg, states_, y, rewards_ = _get_formatted_data(states, actions, rewards, diab, output_state_idx_type)

    return compact_data, emp_tx, emp_r, Xg, states_, y, rewards_

def _get_formatted_data(states, actions, rewards, diabetes, state_index_type):
    Xg = []
    states_ = []
    y = []
    rewards_ = []

    n_steps = states.shape[1] - 1

    for i, trajectory in enumerate(states):
        for t, state_index in enumerate(trajectory):
            state = State(
                state_idx=state_index[0],
                idx_type=state_index_type,
                diabetic_idx=diabetes[i][t][0]  # Ignored if idx_type='full'
            )
            Xg += [state.get_state_vector(idx_type=state_index_type).tolist() + [i]]
            states_ += [state_index]
            y += [actions[i][t][0]]
            rewards_ += [rewards[i][t][0]]

            if t+1 == n_steps or actions[i][t+1] == -1:
                break

    Xg = pd.DataFrame(Xg)
    states_ = np.array(states_)
    y = np.array(y)
    rewards_ = np.array(rewards_)

    return Xg, states_, y, rewards_

# ----------------------------------------------------------
# -- True behavior policy ----------------------------------
# ----------------------------------------------------------

def learn_true_behavior_policy(discount, mdp_params_path):
    tx_mat, r_mat = _load_true_mdp_components(mdp_params_path)

    n_actions = Action.NUM_ACTIONS_TOTAL
    n_full_states = State.NUM_FULL_STATES
    tx_mat_full = np.zeros((n_actions, n_full_states, n_full_states))
    r_mat_full = np.zeros((n_actions, n_full_states, n_full_states))

    for a in range(n_actions):
        tx_mat_full[a, ...] = block_diag(tx_mat[0, a, ...], tx_mat[1, a,...])
        r_mat_full[a, ...] = block_diag(r_mat[0, a, ...], r_mat[1, a, ...])

    full_mdp = cf.MatrixMDP(tx_mat_full, r_mat_full)
    policy, Q = full_mdp.policyIteration(discount=discount, eval_type=1)

    return policy, Q

def _load_true_mdp_components(mdp_params_path):
    with open(mdp_params_path, 'rb') as f:
        mdict = pickle.load(f)
    tx_mat = mdict['tx_mat']
    r_mat = mdict['r_mat']
    return tx_mat, r_mat

# ----------------------------------------------------------
# -- RL target policy --------------------------------------
# ----------------------------------------------------------

def learn_target_policy(obs_samps, emp_tx, emp_r, discount, mdp_params_path, project=False):
    if project:
        assert emp_tx.shape[1] == State.NUM_OBS_STATES
        proj_mat, proj_lookup = _construct_projection_matrix()
    else:
        proj_mat = np.identity(emp_tx.shape[1] + 2)  # Add 2 for absorbing states
        proj_lookup = proj_mat.argmax(axis=-1)

    initial_states = obs_samps[:, 0, 2]
    tx_mat, r_mat, p_initial_state, obs_pol = _construct_mpd_components(emp_tx, emp_r, initial_states, proj_mat)

    mdp = cf.MatrixMDP(tx_mat, r_mat, p_initial_state=p_initial_state)
    rl_pol, Q = mdp.policyIteration(discount=discount)

    ## Check the RL policy

    # Check if we always observe the RL policy in the non-absorbing states
    prop_rl_obs = (obs_pol[:-2, :][rl_pol[:-2, :]==1] > 0).mean()
    if prop_rl_obs < 1:
        assert _check_rl_policy(rl_pol, obs_samps, proj_lookup, mdp_params_path), 'RL policy validation failed'

    return rl_pol, Q, obs_pol

def _construct_projection_matrix():
    '''
    Marginalize out glucose.
    '''
    n_states_abs = State.NUM_OBS_STATES + 2
    n_proj_states = int((State.NUM_OBS_STATES) / 5) + 2
    proj_matrix = np.zeros((n_states_abs, n_proj_states))
    for i in range(n_states_abs - 2):
        this_state = State(state_idx=i, idx_type='obs', diabetic_idx=1)  # Diab a req argument (0/1 does not matter)
        j = this_state.get_state_idx('proj_obs')
        proj_matrix[i, j] = 1
    
    # Add the projection to death and discharge
    proj_matrix[-2, -2] = 1
    proj_matrix[-1, -1] = 1

    proj_matrix = proj_matrix.astype(int)
    proj_lookup = proj_matrix.argmax(axis=-1)

    return proj_matrix, proj_lookup

def _construct_mpd_components(emp_tx_cts, emp_r_mat, initial_states, proj_mat):
    assert emp_tx_cts.ndim == 3

    ## Add absorbing states (death/discharge)

    n_actions = emp_tx_cts.shape[0]
    n_states_abs = emp_tx_cts.shape[1] + 2

    death_state_idx = n_states_abs - 2
    disch_state_idx = n_states_abs - 1

    emp_tx_cts_abs = np.zeros((n_actions, n_states_abs, n_states_abs))
    emp_tx_cts_abs[:, :-2, :-2] = np.copy(emp_tx_cts)  # Should use deep copy?
    
    death_states = (emp_r_mat.sum(axis=0).sum(axis=0) < 0)
    death_states = np.concatenate([death_states, np.array([True, False])])
    
    disch_states = (emp_r_mat.sum(axis=0).sum(axis=0) > 0)
    disch_states = np.concatenate([disch_states, np.array([False, True])])
    
    # We should not transition from states were we observe death/discharge
    assert emp_tx_cts_abs[:, death_states, :].sum() == 0
    assert emp_tx_cts_abs[:, disch_states, :].sum() == 0

    # Instead, we transition to the absorbing states
    emp_tx_cts_abs[:, death_states, death_state_idx] = 1
    emp_tx_cts_abs[:, disch_states, disch_state_idx] = 1

    ## Project emp_tx_cts_abs to a (possibly) reduced state space

    n_proj_states = proj_mat.shape[1]
    proj_tx_cts = np.zeros((n_actions, n_proj_states, n_proj_states))
    for a in range(n_actions):
        proj_tx_cts[a] = proj_mat.T.dot(emp_tx_cts_abs[a]).dot(proj_mat)

    ## Compute the normalized transition matrix

    proj_tx_mat = np.zeros_like(proj_tx_cts)

    nonzero_idx = proj_tx_cts.sum(axis=-1) != 0
    proj_tx_mat[nonzero_idx] = proj_tx_cts[nonzero_idx]
    proj_tx_mat[nonzero_idx] /= proj_tx_mat[nonzero_idx].sum(axis=-1, keepdims=True)

    # Some state-action pairs are never observed -- assume they cause instant death
    zero_sa_pairs = proj_tx_mat.sum(axis=-1) == 0
    proj_tx_mat[zero_sa_pairs, death_state_idx] = 1  # instant death

    ## Estimate the observed policy

    obs_pol_proj = proj_tx_cts.sum(axis=-1)  # Sum over the "to" state
    obs_pol_proj = obs_pol_proj.T  # Switch from (a, s) to (s, a)
    obs_states = obs_pol_proj.sum(axis=-1) > 0  # Observed "from" states
    obs_pol_proj[obs_states] /= obs_pol_proj[obs_states].sum(axis=-1, keepdims=True)

    ## Construct the reward matrix

    proj_r_mat = np.zeros_like(proj_tx_mat)
    
    proj_r_mat[..., death_state_idx] = -1
    proj_r_mat[..., disch_state_idx] = 1

    # No reward once in aborbing state
    proj_r_mat[..., death_state_idx, death_state_idx] = 0
    proj_r_mat[..., disch_state_idx, disch_state_idx] = 0

    ## Construct the initial state distribution

    initial_state_counts = np.zeros((n_states_abs, 1))
    for initial_state in initial_states:
        initial_state_counts[int(initial_state)] += 1

    proj_state_counts = proj_mat.T.dot(initial_state_counts).T

    proj_p_initial_state = proj_state_counts / proj_state_counts.sum()

    return proj_tx_mat, proj_r_mat, proj_p_initial_state, obs_pol_proj

def _check_rl_policy(rl_policy, obs_samps, proj_lookup, mdp_params_path):
    '''
    This is a QA function to ensure that the RL policy is only taking actions that have been observed.
    '''
    passes = True
    
    n_samples = obs_samps.shape[0]
    n_steps = obs_samps.shape[1]

    _, r_mat = _load_true_mdp_components(mdp_params_path)
    
    # Check the observed actions for each state
    obs_pol = np.zeros_like(rl_policy)
    for eps_idx in range(n_samples):
        for time_idx in range(n_steps):
            this_obs_action = int(obs_samps[eps_idx, time_idx, 1])
            if this_obs_action == -1:
                continue
            # Need to get projected state
            this_obs_state = proj_lookup[int(obs_samps[eps_idx, time_idx, 2])]
            obs_pol[this_obs_state, this_obs_action] += 1
    
    # Check if each RL action conforms to an observed action
    for eps_idx in range(n_samples):
        for time_idx in range(n_steps):
            this_full_state_unobserved = int(obs_samps[eps_idx, time_idx, 1])  # State = Action??? (Action has index 1 in obs_samps.)
            this_obs_state = proj_lookup[this_full_state_unobserved]
            this_obs_action = int(obs_samps[eps_idx, time_idx, 1])
            if this_obs_action == -1:
                continue
            # This is key: In some of these trajectories, you die or get discharge.  
            # In this case, no action is taken because the sequence has terminated, so there's nothing to compare the RL action to.
            true_death_states = r_mat[0, 0, 0, :] == -1
            true_disch_states = r_mat[0, 0, 0, :] == 1
            if np.logical_or(true_death_states, true_disch_states)[this_full_state_unobserved]:
                continue
            this_rl_action = rl_policy[proj_lookup[this_obs_state]].argmax()
            if obs_pol[this_obs_state, this_rl_action] == 0:
                print('Eps: {} \t RL Action {} in State {} never observed'.format(
                    int(time_idx / n_steps), this_rl_action, this_obs_state)
                )
                passes = False
    
    return passes
