import os
import pickle
import torch
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from skorch.exceptions import DeviceWarning

# ----------------------------------------------------------
# -- Helpers -----------------------------------------------
# ----------------------------------------------------------

def extract_treatments(y):
    fluid = []
    vp = []
    for y_ in y:
        fluid += [int(y_/5)]
        vp += [y_%5]
    return fluid, vp

def get_formatted_treatment(y):
    if isinstance(y, str):
        y = int(y)
    f, v = extract_treatments([y])
    return '({}, {})'.format(f[0], v[0])

def model_compute(model, X):
    from skorch.dataset import unpack_data

    transformer = model.base_estimator[0]
    net = model.base_estimator[-1]

    X_tf = transformer.transform(X)
    dataset = net.get_dataset(X_tf)

    outputs = []
    similarities = []
    encodings = []
    
    for data in net.get_iterator(dataset, training=True):
        Xi = unpack_data(data)[0]
        with torch.no_grad():
            net.module_.eval()
            batch_outputs, batch_similarities, batch_encodings = net.infer(Xi)
        outputs += [batch_outputs]
        similarities += [batch_similarities]
        encodings += [batch_encodings]
    
    outputs = np.concatenate(outputs, 0)
    similarities = np.concatenate(similarities, 0)
    encodings = np.concatenate(encodings, 0)

    return outputs, similarities, encodings

def compute_ratios_weights(probas_pi, probas_mu, actions, sequence_indices):
    ratios, weights = [], []
    for ii_sequence in sequence_indices:
        actions_sequence = actions[ii_sequence]
        probas_pi_ = [[probas_pi[i, a] for i, a in zip(ii_sequence, actions_sequence)]]
        probas_mu_ = [[probas_mu[i, a] for i, a in zip(ii_sequence, actions_sequence)]]
        ratios += np.divide(probas_pi_, probas_mu_).tolist()
        weights += [np.prod(ratios[-1])]
    return ratios, np.array(weights)

def remove_outlier_sequences(sequence_indices, weights, threshold, verb=0):
    import copy
    sequence_indices_no_outliers = copy.deepcopy(sequence_indices)

    ii_outliers = np.nonzero(weights>threshold)[0]
    
    for i_outlier in np.flip(ii_outliers):
        if verb > 0:
            print(sequence_indices_no_outliers.pop(i_outlier))
        else:
            sequence_indices_no_outliers.pop(i_outlier)

    return sequence_indices_no_outliers, len(ii_outliers)

# ----------------------------------------------------------
# -- Data loading ------------------------------------------
# ----------------------------------------------------------

def load_value_estimates(values):
    mus = []
    pis = []
    estimates = []

    for pi, d in values.items():
        for mu, estimates_ in d.items():
            n = len(estimates_)
            mus += n*[mu]
            pis += n*[pi]
            estimates += estimates_

    return pd.DataFrame({'mu': mus, 'pi': pis, 'estimate': estimates})

def get_model_paths_from_directory(dir_path):
    model_paths = {}
    for file in os.listdir(dir_path):
        if file.startswith('lr'):
            model = 'lr'
        elif file.startswith('rf'):
            model = 'rf'
        elif file.startswith('mlp'):
            model = 'mlp'
        elif file.startswith('rnn'):
            model = 'rnn'
        elif file.startswith('pro'):
            model = file[:-len('.pickle')].split('_')
            if 'cal' in model:
                model.pop(1)  # Ignore 'cal'
            model = '_'.join(model)
        else:
            continue
        model_paths[model] = os.path.join(dir_path, file)
    return model_paths

def load_models(model_paths):
    models = {}
    for model_name, model_path in model_paths.items():
        if model_path is None:
            continue
        with open(model_path, 'rb') as f:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeviceWarning)
                model = pickle.load(f)
        models[model_name] = model
    return models

# ----------------------------------------------------------
# -- Prototype performance ---------------------------------
# ----------------------------------------------------------

def compute_prototype_accuracies(mu_models, Xg_test, y_test):
    from sklearn.metrics import accuracy_score
    
    df = {
        'Model': [],
        'Accuracy': [],
        '# Prototypes': [],
        '# Prediction prototypes': []
    }
    
    for mu, mu_model in mu_models.items():
        if not mu.startswith('pro'):
            continue

        df['Model'] += ['ProNet'] if mu.startswith('pronet') else ['ProSeNet']
        df['Accuracy'] += [accuracy_score(y_test, mu_model.predict(Xg_test))]
        
        net = mu_model.base_estimator[-1]
        df['# Prototypes'] += [net.module_.n_prototypes]
        df['# Prediction prototypes'] += [net.n_prediction_prototypes]
    
    return pd.DataFrame(df)

def compute_prototype_similarities(mu_models, Xg_test, n_similar=5):
    df = {
        'Model': [],
        'Similarity': [],
        'Prototype': [],
        '# Prototypes': [],
        '# Prediction prototypes': []
    }
    
    for mu, mu_model in mu_models.items():
        if not mu.startswith('pro'):
            continue
        
        df['Model'] += n_similar*['ProNet'] if mu.startswith('pronet') else n_similar*['ProSeNet']
        
        net = mu_model.base_estimator[-1]
        df['# Prototypes'] += n_similar*[net.module_.n_prototypes]
        df['# Prediction prototypes'] += n_similar*[net.n_prediction_prototypes]
        
        _, similarities, _ = model_compute(mu_model, Xg_test)
        top_similarities = np.sort(similarities, axis=1)[:, -n_similar:].mean(axis=0)
        df['Similarity'] += list(np.flip(top_similarities))
        df['Prototype'] += [str(i) for i in range(1, n_similar+1)]
    
    return pd.DataFrame(df)

# ----------------------------------------------------------
# -- Visualizations ----------------------------------------
# ----------------------------------------------------------

def visualize_encodings(model, X_train, hue, hue_levels, frac=0.1, annotations=None, figsize=(6,4)):
    from sklearn.decomposition import PCA

    _, _, encodings = model_compute(model, X_train)
    pca = PCA(n_components=2)
    encodings_pca = pca.fit_transform(encodings)
    
    df = pd.DataFrame(
        {
            'PC 1': encodings_pca[:, 0], 
            'PC 2': encodings_pca[:, 1], 
            'Prototype': encodings.shape[0] * ['No'],
            hue: hue_levels
        }
    )
    df = df.sample(frac=frac, axis='index')
    
    net = model.base_estimator[-1]
    if net.module_.should_use_prototypes:
        prototypes = net.module_.prototype_layer.prototypes.detach().cpu().numpy()
        prototypes_pca = pca.transform(prototypes)
        n_prototypes = prototypes.shape[0]
        ii_prototypes = net.history[-1]['prototype_indices']
        df_prototypes = pd.DataFrame(
            {
                'PC 1': prototypes_pca[:, 0], 
                'PC 2': prototypes_pca[:, 1], 
                'Prototype': n_prototypes * ['Yes'],
                hue: hue_levels[ii_prototypes]
            }
        )
    else:
        df_prototypes = None
    
    fig, ax = plt.subplots(figsize=figsize)
    common_kwargs = {'x': 'PC 1', 'y': 'PC 2', 'hue': hue, 'ax': ax}
    sns.scatterplot(
        data=df,
        alpha=0.7,
        size='Prototype',
        sizes=(20, 100),
        size_order=['Yes', 'No'],
        **common_kwargs
    )
    if df_prototypes is not None: 
        sns.scatterplot(
            data=df_prototypes,
            alpha=1,
            s=n_prototypes*[100],
            legend=False, 
            **common_kwargs
        )
        # Annotate prototypes
        for i, a in enumerate(prototypes_pca, start=1):
            try:
                xytext = (a[0]+annotations[i][0], a[1]+annotations[i][1])
                ax.annotate(i, xy=a, xytext=xytext, arrowprops={'arrowstyle': '-'})
            except TypeError:
                ax.annotate(i, xy=(a[0]+0.2, a[1]))
    ax.legend(loc='center right', bbox_to_anchor=(1.35, 0.5))

    return fig, ax

def visualize_prototypes_sepsis(Xg_train, y_train, c_features, prototype_indices, prototype_numbers, figsize):
    from matplotlib.ticker import MaxNLocator
    
    assert len(prototype_indices) <= 4
    assert isinstance(Xg_train.index, pd.RangeIndex)
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', 'D', 'P']
    
    n_features = len(c_features)
    n_prototypes = len(prototype_indices)
    
    fig, axes = plt.subplots(n_features+2, 1, sharex=True, sharey=False, figsize=figsize)
    axes = axes.flatten()
    
    for ax, c_feature in zip(axes, c_features):
        avg = Xg_train[c_feature].mean()
        std = Xg_train[c_feature].std()
        
        ax.hlines(y=avg, xmin=0, xmax=19, color='black', linestyle='dotted')
        ax.fill_between(range(0, 20), 20*[avg-std], 20*[avg+std], color='black', alpha=0.1)
        
        ax.set_xlim([-0.5, 19.5])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim([avg-3*std, avg+3*std])
        ax.set_ylabel(c_feature)
        for label in ax.get_yticklabels():
            label.set_rotation(90)
        
        i_c_feature = Xg_train.columns.get_loc(c_feature)
        for i_prototype, color, marker in zip(prototype_indices, colors, markers):
            sequence = Xg_train[Xg_train.iloc[:, -1] == Xg_train.iloc[i_prototype, -1]]
            ts = range(len(sequence.index))
            t_prototype = list(sequence.index).index(i_prototype)
            ax.plot(ts[:t_prototype+1], sequence.iloc[:t_prototype+1, i_c_feature], c=color)
            ax.scatter(ts[t_prototype], sequence.iloc[t_prototype, i_c_feature], c=color, marker=marker)
            ax.plot(ts[t_prototype:], sequence.iloc[t_prototype:, i_c_feature], linestyle='dashed', c=color)
    
    artists = [axes[0].plot([], [], markers[i]+'-.', color=colors[i])[0] for i in range(n_prototypes)]
    labels = ['Prototype {}'.format(i) for i in prototype_numbers]
    axes[0].legend(artists, labels, loc='lower right')
    
    for i_prototype, color, marker in zip(prototype_indices, colors, markers):
        sequence = Xg_train[Xg_train.iloc[:, -1] == Xg_train.iloc[i_prototype, -1]]
        ts = range(len(sequence.index))
        t_prototype = list(sequence.index).index(i_prototype)
        y_sequence = y_train[sequence.index]
        treatments = extract_treatments(y_sequence)
        for ax, treatments_ in zip(axes[n_features:], treatments):
            ax.plot(ts[:t_prototype+1], treatments_[:t_prototype+1], color=color)
            ax.scatter(ts[t_prototype], treatments_[t_prototype], c=color, marker=marker)
            ax.plot(ts[t_prototype:], treatments_[t_prototype:], linestyle='dashed', color=color)
    
    axes[-2].set_ylabel('IV fluids')
    axes[-1].set_ylabel('Vasopressors')
    axes[-1].set_xlabel('Time step')
    for ax in axes[-2:]:
        ax.grid(True, axis='y')
        ax.set_ylim([-0.5, 4.5])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    return fig, axes

def compare_action_distributions_sepsis_new(probas_mu, probas_pi, prototype_numbers, broken_axis, figsize, colors=None):
    def break_axis(axes, ymax_lower=0.225, ymin_upper=0.725):
        ax1, ax2 = axes
        
        ax1.set_ylim([ymin_upper, 1])
        ax2.set_ylim([0, ymax_lower])

        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        
        ax1.tick_params(bottom=False, labelbottom=False, top=False, labeltop=False)
        ax2.tick_params(top=False)

        d = 0.005
        s = 0.001
        
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)  # Start with top axes
        ax1.plot((-d, +d), (-d-s, +d-s), linewidth=0.8, **kwargs)  # Top-left diagonal
        ax1.plot((1-d, 1+d), (-d-s, +d-s), linewidth=0.8, **kwargs)  # Top-right diagonal

        kwargs.update(transform=ax2.transAxes)  # Switch to the bottom axes
        ax2.plot((-d, +d), (1-d+s, 1+d+s), linewidth=0.8, **kwargs)  # Bottom-left diagonal
        ax2.plot((1-d, 1+d), (1-d+s, 1+d+s), linewidth=0.8, **kwargs)  # Bottom-right diagonal
        
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    x = np.arange(25)
    xticklabels = [get_formatted_treatment(i) if i%5==0 else '' for i in range(25)]

    w = 0.375
    s = 0.05
    
    n_prototypes = len(prototype_numbers)
    n_rows = n_prototypes
    fig, axes = plt.subplots(nrows=n_rows, sharex=True, figsize=figsize)

    if n_rows == 1: axes = [axes]
    
    for i, (probas_mu_, probas_pi_, color) in enumerate(zip(probas_mu, probas_pi, colors)):
        ax1 = axes[i]
        if broken_axis[i]:        
            divider = make_axes_locatable(ax1)
            ax2 = divider.new_vertical(size='100%', pad=0.1)
            fig.add_axes(ax2)
            axes_ = [ax1, ax2]
        else:
            axes_ = [ax1]
        
        prototype_number = prototype_numbers[i]
        mu_label = r'Physicians ($\mu$)'
        pi_label = r'AI Clinician ($\pi$)'
        
        for ax in axes_:
            ax.set_xlim([-1, 25])
            ax.bar(x-w/2-s/2, probas_mu_, w, color=color, label=mu_label)
            ax.bar(x+w/2+s/2, probas_pi_, w, hatch='//', label=pi_label, color='white', edgecolor=color, zorder=0)
        
        if broken_axis[i]:
            if isinstance(broken_axis[i], tuple):
                break_axis((ax2, ax1), *broken_axis[i])
            else:
                break_axis((ax2, ax1))
            ax1.set_ylabel('Probability', y=1.1, horizontalalignment='center')
            ax2.legend(loc='best', title='Prototype {}'.format(prototype_number))
        else:
            ax1.set_ylabel('Probability')
            ax1.legend(loc='upper right', title='Prototype {}'.format(prototype_number))
        
        if i+1 == n_rows:
            ax1.set_xlabel('Action (f, v)')
            ax1.set_xticks(x)
            ax1.set_xticklabels(xticklabels, rotation=90)
    
    return fig, axes

def compare_action_distributions_sepsis(probas_mu, probas_pi, prototype_numbers, figsize):    
    def _break_axis(axes, ylim_lower=[0, 0.275], ylim_upper=[0.725, 1]):
        ax1, ax2 = axes
        
        ax1.set_ylim(ylim_upper)
        ax2.set_ylim(ylim_lower)

        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        
        ax1.tick_params(bottom=False, labelbottom=False, top=False, labeltop=False)
        ax2.tick_params(top=False)

        d = 0.005
        s = 0.001
        
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)  # Start with top axes
        ax1.plot((-d, +d), (-d-s, +d-s), linewidth=0.8, **kwargs)  # Top-left diagonal
        ax1.plot((1-d, 1+d), (-d-s, +d-s), linewidth=0.8, **kwargs)  # Top-right diagonal

        kwargs.update(transform=ax2.transAxes)  # Switch to the bottom axes
        ax2.plot((-d, +d), (1-d+s, 1+d+s), linewidth=0.8, **kwargs)  # Bottom-left diagonal
        ax2.plot((1-d, 1+d), (1-d+s, 1+d+s), linewidth=0.8, **kwargs)  # Bottom-right diagonal
        
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import MaxNLocator
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    x = np.arange(25)
    xticklabels = [get_formatted_treatment(i) if i%5==0 else '' for i in range(25)]
    
    w = 0.4
    s = 0.05
    
    n_prototypes = len(prototype_numbers)
    n_rows = int(np.ceil(n_prototypes / 2))
    fig, axes = plt.subplots(nrows=n_rows, sharex=True, figsize=figsize)
    
    if n_rows == 1: axes = [axes]
    
    for i, (probas_mu_, probas_pi_, color) in enumerate(zip(probas_mu, probas_pi, colors)):        
        if i%2 == 0:
            ax1 = axes[int(i/2)]
            divider = make_axes_locatable(ax1)
            ax2 = divider.new_vertical(size='100%', pad=0.1)
            fig.add_axes(ax2)
            axes_ = [ax1, ax2]
        
        prototype_number = prototype_numbers[i]
        mu_label = '%d: Behavior policy' % prototype_number
        pi_label = '%d: Target policy' % prototype_number
        
        for ax in axes_:
            ax.set_xlim([-1, 25])
            x_ = x-w/2-s/2 if i%2 == 0 else x+w/2+s/2
            ax.bar(x_, probas_mu_, w, color=color, label=mu_label)
            ax.bar(x_, probas_pi_, w, hatch='//', label=pi_label, color='white', edgecolor=color, zorder=0)
        
        if i%2 == 1 or i+1 == n_prototypes:
            _break_axis((ax2, ax1))
            
            ax1.set_ylabel('Probability', y=1.1, horizontalalignment='center')
            
            if i+1 == n_prototypes:
                ax1.set_xlabel('Action (f, v)')
                if i%2 == 1:
                    ax2.legend(loc='best', title='Prototype {} \& {}'.format(prototype_numbers[i-1], prototype_number))
                else:
                    ax2.legend(loc='best', title='Prototype {}'.format(prototype_number))
            else:
                ax2.legend(loc='best', title='Prototype {} \& {}'.format(prototype_numbers[i-1], prototype_number))
            
            ax1.set_xticks(x)
            ax1.set_xticklabels(xticklabels, rotation=90)
    
    return fig, axes

# ----------------------------------------------------------
# -- Prototype-based value contributions -------------------
# ----------------------------------------------------------

def compute_Vt(t, rewards, probas_pi, probas_mu, actions, similarities, sequence_indices, use_wis=True):    
    n_sequences = len(sequence_indices)
    
    ratios = []
    final_rewards = []
    similarities_ = []
    
    for ii_sequence in sequence_indices:
        if len(ii_sequence) > t:
            actions_sequence = actions[ii_sequence]
            probas_pi_ = [[probas_pi[i, a] for i, a in zip(ii_sequence, actions_sequence)]]
            probas_mu_ = [[probas_mu[i, a] for i, a in zip(ii_sequence, actions_sequence)]]
            ratios += [np.divide(probas_pi_, probas_mu_)]
            
            rewards_sequence = rewards[ii_sequence]
            final_rewards += [rewards_sequence[-1]]
            
            similarities_sequence = similarities[ii_sequence]
            similarities_ += [similarities_sequence[t]]
    
    final_rewards = np.array(final_rewards)
    similarities_ = np.array(similarities_)
    
    def similarities_to_probas(s): 
        return s / np.sum(s)
    similarity_probas = np.apply_along_axis(similarities_to_probas, axis=1, arr=similarities_)
    
    wt = np.array([np.prod(ratios_[:t]) for ratios_ in ratios])
    wT = np.array([np.prod(ratios_) for ratios_ in ratios])
    
    inner_norm = np.sum(wt) if use_wis else n_sequences
    inner_sum = np.sum(similarity_probas * wt[:, None], axis=0) / inner_norm
    inv_inner_sum = 1 / inner_sum
    w_tilde = (inv_inner_sum * similarity_probas) * wT[:, None]
    
    n_e = np.square(np.sum(w_tilde, axis=0)) / np.sum(np.square(w_tilde), axis=0)
    
    outer_norm = np.sum(wT) if use_wis else n_sequences
    V_t = np.sum(w_tilde * final_rewards[:, None], axis=0) / outer_norm
    
    pJt = inner_sum
    assert np.sum(pJt) > 0.999 and np.sum(pJt) < 1.001
    
    V = np.sum(V_t * pJt)
    
    return V_t, n_e, pJt, V
