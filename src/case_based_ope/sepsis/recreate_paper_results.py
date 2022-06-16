import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from case_based_ope.sepsis.data import Data
from case_based_ope.utils.misc import load_config
from matplotlib.ticker import MaxNLocator
from case_based_ope.sepsis.target_policies import *
from case_based_ope.utils.postprocessing import visualize_encodings
from case_based_ope.utils.postprocessing import get_formatted_treatment
from case_based_ope.utils.postprocessing import get_model_paths_from_directory
from case_based_ope.utils.postprocessing import load_models
from case_based_ope.utils.postprocessing import compute_prototype_accuracies
from case_based_ope.utils.postprocessing import visualize_prototypes_sepsis
from case_based_ope.utils.postprocessing import model_compute
from case_based_ope.utils.postprocessing import compare_action_distributions_sepsis_new
from case_based_ope.utils.postprocessing import load_value_estimates
from case_based_ope.utils.postprocessing import compute_ratios_weights
from case_based_ope.utils.postprocessing import remove_outlier_sequences
from case_based_ope.utils.postprocessing import compute_Vt
from case_based_ope.utils.evaluation import get_scores
from pathlib import Path
from case_based_ope.utils.misc import get_args

def main(config):
    figures_path = './paper/figures/'
    Path(figures_path).mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
            'figure.titlesize': 16,
            'axes.titlesize': 16,
            'axes.labelsize': 16,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'font.size': 13,
            'legend.title_fontsize': 13,
            'legend.fontsize': 13,
        }
    )

    # ----------------------------------------------------------
    # -- Loading data, models and results ----------------------
    # ----------------------------------------------------------

    train_data, valid_data, test_data = Data.get_train_valid_test_data(
        config.data.path,
        config.ai_clinician.path,
        config.data.valid_size,
        config.data.test_size,
        config.data.use_aic_split,
        config.data.seed
    )

    Xg_train, y_train, states_train, _ = train_data
    Xg_test, y_test, states_test, rewards_test = test_data

    Xg_valid, y_valid, _, _ = valid_data

    Xg_train.reset_index(drop=True, inplace=True)
    Xg_test.reset_index(drop=True, inplace=True)

    sequence_indices_train = [group[1].index.tolist() for group in Xg_train.groupby(by=Data.C_GROUP)]
    sequence_indices_test = [group[1].index.tolist() for group in Xg_test.groupby(by=Data.C_GROUP)]

    model_dir = join(config.results.path, 'models/')
    model_paths = get_model_paths_from_directory(model_dir)
    mu_models = load_models(model_paths)

    with open(join(config.results.path, 'scores.pickle'), 'rb') as f:
        scores = pickle.load(f)

    with open(join(config.results.path, 'wis_estimates.pickle'), 'rb') as f:
        wis_estimates = pickle.load(f)

    # ----------------------------------------------------------
    # -- Evaluating the AI Clinician ---------------------------
    # ----------------------------------------------------------

    mu_model_str = 'prosenet_10_2_5'

    mu_model = mu_models[mu_model_str]
    mu_net = mu_model.base_estimator[-1]
    ii_prototypes = mu_net.history[-1]['prototype_indices']

    similarities_train = model_compute(mu_model, Xg_train)[1]
    similarities_test = model_compute(mu_model, Xg_test)[1]

    pi_aic = AICPolicy(join(config.ai_clinician.path, 'target_policy.csv'))
    pi_zero = ZeroDrugPolicy(p=0)

    probas_mu_train = mu_model.predict_proba(Xg_train)
    probas_mu_test = mu_model.predict_proba(Xg_test)

    #probas_aic_train = pi_aic.predict_proba(states_train)
    probas_aic_test = pi_aic.predict_proba(states_test)

    #probas_zero_train = pi_zero.predict_proba(states_train)
    probas_zero_test = pi_zero.predict_proba(states_test)

    # ----------------------------------------------------------
    # -- Overviewing the prototypes  ---------------------------
    # ----------------------------------------------------------

    fig, ax = visualize_encodings(
        mu_model,
        Xg_train,
        hue='Action',
        hue_levels=y_train,
        annotations={
            1: (0.3, -0.23),
            2: (-0.4, -0.3),
            3: (0.32, -0.27),
            4: (0.4, 0.22),
            5: (0, -0.7),
            6: (-0.4, -0.3),
            7: (0.3, -0.23),
            8: (-0.4, -0.3),
            9: (-0.45, 0.2),
            10: (-0.6, 0.25)
        },
        figsize=(4, 4),
        frac=0.01
    )

    current_handles, current_labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in current_labels:
        if label == 'Action':
            new_label = 'Action\n(f, v)'
        elif 'mathdefault' in label:
            y = re.findall(r'\d+', label)[0]
            new_label = get_formatted_treatment(y)
        else:
            new_label = label
        new_labels += [new_label]
    ax.legend(current_handles, new_labels, loc='center right', bbox_to_anchor=(1.55, 0.5));

    fig.savefig(join(figures_path, '{}_pca.pdf'.format(mu_model_str)), bbox_inches='tight', dpi=300)

    # ----------------------------------------------------------
    # -- Interpreting prototype 5, 7 and 8 ---------------------
    # ----------------------------------------------------------

    prototypes = [5, 7, 8]

    fig, axes = visualize_prototypes_sepsis(
        Xg_train,
        y_train,
        c_features=['HR', 'MeanBP', 'SOFA'],
        prototype_indices=ii_prototypes[[p-1 for p in prototypes]],
        prototype_numbers=prototypes,
        figsize=(4.5, 7)
    )
    axes[1].set_ylabel('Mean BP');

    fig.savefig(join(figures_path, '{}_features_actions_{}.pdf'.format(mu_model_str, ''.join([str(p) for p in prototypes]))), bbox_inches='tight', dpi=300)

    # ----------------------------------------------------------
    # -- Inspecting prototype 7 and 9 --------------------------
    # ----------------------------------------------------------

    probas_mu_prototypes = probas_mu_train[ii_prototypes]

    n_prototypes = len(ii_prototypes)
    n_actions = len(set(y_train))
    probas_aic_prototypes = np.zeros((n_prototypes, n_actions))
    time_index_train = np.array([t for ii in sequence_indices_train for t in range(len(ii))])
    for p in range(n_prototypes):
        # Select patients who are most similar to prototype p at the first time point
        mask = (similarities_train.argmax(axis=1) == p) & (time_index_train == 0)
        if np.sum(mask) > 0:
            probas_aic_prototypes[p, :] = np.bincount(pi_aic.predict_proba(states_train[mask]).argmax(axis=1), minlength=n_actions)
    with np.errstate(invalid='ignore'):
        probas_aic_prototypes = np.array(probas_aic_prototypes) / np.sum(probas_aic_prototypes, axis=1)[:, np.newaxis]
    
    prototypes = [7, 9]

    fig, axes = compare_action_distributions_sepsis_new(
        probas_mu_prototypes[[p-1 for p in prototypes]],
        probas_aic_prototypes[[p-1 for p in prototypes]],
        prototypes,
        broken_axis=2*[False],
        figsize=(6, 6),
        colors=['tab:orange', 'tab:red']
    )

    fig.savefig(join(figures_path, 'probas_{}.pdf'.format(''.join([str(p) for p in prototypes]))), bbox_inches='tight', dpi=300)

    # ----------------------------------------------------------
    # -- Comparing policy value estimates ----------------------
    # ----------------------------------------------------------

    wis_values_df = load_value_estimates(wis_estimates)
    wis_values_df.replace(
        to_replace={'aic': 'AI Clinician', 'zero_drug': 'Zero-drug'},
        inplace=True
    )
    wis_values_df = wis_values_df[wis_values_df.mu.isin(['lr', 'rf', 'mlp', 'rnn', mu_model_str])]

    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.boxplot(
            x='pi', 
            y='estimate', 
            hue='mu', 
            data=wis_values_df, 
            ax=ax,
            hue_order=['lr', 'rf', 'mlp', 'rnn', mu_model_str]
        )
        ax.set_xlabel('Initial policy')
        ax.set_ylabel('$\hat{V}_{\mathrm{WIS}}(\pi)$')
        ax.legend(loc='center right', bbox_to_anchor=(1.45, 0.5), title='Model')
        
    current_handles, current_labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in current_labels:
        if label == 'lr':
            new_label = 'LR'
        elif label == 'rf':
            new_label = 'RF'
        elif label == 'mlp':
            new_label = 'FNN'
        elif label == 'rnn':
            new_label = 'RNN'
        elif label == mu_model_str:
            new_label = 'Ours'
        else:
            new_label = label
        new_labels += [new_label]
    ax.legend(current_handles, new_labels, loc='center right', bbox_to_anchor=(1.38, 0.5), title='Model');

    V_mu = np.sum(rewards_test) / len(sequence_indices_test)
    ax.scatter([0.5], [V_mu], c=['black'], marker='*', s=[150]);
    ax.annotate(r'$\hat{V}(\mu)$', xy=(0.5, V_mu), xytext=(0.6, 40), ha='center', arrowprops=dict(arrowstyle='-'));
    ax.set_xlim(-0.5, 1.5);
    ax.set_ylim(30, 70)

    fig.savefig(join(figures_path, 'policy_values.pdf'), bbox_inches='tight', dpi=300)

    # ----------------------------------------------------------
    # -- Computing value per prototype -------------------------
    # ----------------------------------------------------------

    t_value = 0  # Compute prototype-based value contributions at the first time step
    prototypes = [1, 7, 9]

    Vt_data = {'policy': [], 'time': [], 'prototype': [], 'Vt': [], 'VtpJt': []}

    for pi, probas_pi in {'aic': probas_aic_test, 'zero': probas_zero_test, 'mu': probas_mu_test}.items():
        probas_pi_mod = np.zeros_like(probas_pi)
        if not config.pi_until is None:
            for ii_sequence in sequence_indices_test:
                first_probas = probas_pi[ii_sequence[:config.pi_until]]
                second_probas = probas_mu_test[ii_sequence[config.pi_until:]]
                probas_pi_mod[ii_sequence] = np.vstack((first_probas, second_probas))
        else:
            probas_pi_mod = probas_pi
        
        _, weights = compute_ratios_weights(probas_pi_mod, probas_mu_test, y_test, sequence_indices_test)
        sequence_indices, _ = remove_outlier_sequences(sequence_indices_test, weights, threshold=config.weight_threshold)
        
        n_sequences = len(sequence_indices)
        bootstrap_indices = np.random.randint(low=0, high=n_sequences, size=(config.n_is_bootstraps, n_sequences))
        
        for ii_bootstrap in bootstrap_indices:
            sequence_indices_ = [sequence_indices[i] for i in ii_bootstrap]
            Vt, _, pJt, _ = compute_Vt(
                t_value,
                rewards_test,
                probas_pi_mod,
                probas_mu_test,
                y_test,
                similarities_test,
                sequence_indices_
            )
        
            for j, Vt_j in enumerate(Vt, start=1):
                Vt_data['policy'] += [pi]
                Vt_data['time'] += [t_value]
                Vt_data['prototype'] += [j]
                Vt_data['Vt'] += [Vt_j]
                Vt_data['VtpJt'] += [Vt_j*pJt[j-1]]
    
    Vt_df = pd.DataFrame(Vt_data)

    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(figsize=(4.5, 3))
        sns.boxplot(
            data=Vt_df[(Vt_df['time']==t_value) & (Vt_df['prototype'].isin(prototypes))],
            x='prototype',
            y='VtpJt',
            hue='policy',
            ax=ax,
            linewidth=0.75
        )
        ax.set_xlabel(r'Prototype ($j$)');
        if t_value == 0:
            ylabel = '$\hat{V}_{j, {-}}(\pi)\hat{p}(J_{-}=j)$'.replace('-', str(t_value))
        else:
            ylabel = '$\hat{V}_{j, {-}}(\pi)\hat{p}_{\pi}(J_{-}=j)$'.replace('-', str(t_value))
        ax.set_ylabel(ylabel);

        current_handles, current_labels = ax.get_legend_handles_labels()
        new_labels = []
        for label in current_labels:
            if label == 'aic':
                new_label = 'AI Clinician'
            elif label == 'zero':
                new_label = 'Zero-drug'
            elif label == 'mu':
                new_label = 'Physicians'
            else:
                new_label = label
            new_labels += [new_label]

        ax.legend(
            current_handles,
            new_labels,
            loc='best',
            ncol=1,
            title='Initial policy',
        );

        ax.set_ylim([0, 30]);

    fig.savefig(join(figures_path, 'VtpJt_{}.pdf'.format(t_value)), bbox_inches='tight', dpi=300)

    # ----------------------------------------------------------
    # -- ProNet vs ProSeNet ------------------------------------
    # ----------------------------------------------------------

    df_accuracy = compute_prototype_accuracies(mu_models, Xg_test, y_test)
    df_accuracy_ = df_accuracy.rename(
        mapper={
            '# Prediction prototypes': r'\# Prediction prototypes ($q$)',
            '# Prototypes': r'\# Prototypes ($n$)',
        },
        axis=1
    )

    fig, ax = plt.subplots(figsize=(4.5, 3))
    sns.lineplot(
        data=df_accuracy_,
        x='\# Prediction prototypes ($q$)',
        y='Accuracy',
        hue='\# Prototypes ($n$)',
        style='Model',
        markers=True,
        dashes=True,
        ax=ax,
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True));
    fig.savefig(join(figures_path, 'pronet_prosenet_acc.pdf'), bbox_inches='tight', dpi=300)

    # ----------------------------------------------------------
    # -- Prototype models vs baselines -------------------------
    # ----------------------------------------------------------

    def print_scores(model_scores):
        for model, scores in model_scores.items():
            print(model)
            for metric, score in scores.items():
                if metric == 'sce':
                    strformat = '  %s: %.4f (%.4f, %.4f)'
                else:
                    strformat = '  %s: %.2f (%.2f, %.2f)'
                formatted = strformat % (metric, score[0], score[1], score[2])
                print(formatted)
            print()

    pronet = 'pronet_10_2_5'
    prosenet = 'prosenet_10_2_5'
    baselines = ['lr', 'rf', 'mlp', 'rnn', 'posthoc']

    models = [pronet, prosenet] + baselines

    print_scores({m: scores[m] for m in models})

if __name__ == '__main__':
    args = get_args()
    config_path = args.config_path
    config = load_config(config_path)
    main(config)