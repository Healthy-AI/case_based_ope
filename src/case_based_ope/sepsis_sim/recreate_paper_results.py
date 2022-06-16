import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from os.path import join
from case_based_ope.utils.misc import load_config
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

    wis_data = pd.read_csv(join(config.results.path, 'wis_data.csv'))

    # ----------------------------------------------------------
    # -- Bias as a function of sequence length -----------------
    # ----------------------------------------------------------

    models = ['mlp', 'pronet_10_2', 'pronet_10_5', 'pronet_100_5']
    wis_data_ = wis_data[wis_data.model.isin(models)]

    wis_data_no_outliers = wis_data_[wis_data_['w_ratio'] < config.w_ratio_th]
    #print('Removed {} out of {} samples.'.format(wis_data_.shape[0]-wis_data_no_outliers.shape[0], wis_data_.shape[0]))
    
    wis_data_no_outliers = wis_data_no_outliers.assign(
        relative_w_ratio=wis_data_no_outliers['w_ratio'].transform(lambda x: np.abs(1-x)).values
    )
    
    fig, ax = plt.subplots(figsize=(5, 4))

    sns.lineplot(
        x='time',
        y='relative_w_ratio',
        hue='model',
        hue_order=['mlp', 'pronet_10_2', 'pronet_10_5', 'pronet_100_5'],
        data=wis_data_no_outliers,
        ax=ax,
        ci=95
    )

    ax.set_xlabel('Sequence length');
    ax.set_ylabel(r'$\left|1-w_{\hat{\mu}}/w_{\mu}\right|$');

    current_handles, current_labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in current_labels:
        if label == 'mlp':
            new_label = 'FNN'
        elif label == 'pronet_10_2':
            new_label = 'ProNet\n($n$=10, $q$=2)'
        elif label == 'pronet_10_5':
            new_label = 'ProNet\n($n$=10, $q$=5)'
        elif label == 'pronet_100_5':
            new_label = 'ProNet\n($n$=100, $q$=5)'
        else:
            new_label = label
        new_labels += [new_label]
    ax.legend(
        current_handles,
        new_labels,
        ncol=1,
        title=r'Estimator of $\mu$',
        loc='center right', 
        bbox_to_anchor=(1.5, 0.5)
    );

    ax.axvline(x=13.25, color='black', linestyle='dashed', linewidth=1);
    ax.annotate(
        'Average\nsequence\nlength in\nMIMIC-III\ndata',
        xy=(13.25, 4),
        xytext=(15, 4),
        arrowprops={'arrowstyle': '->'},
        horizontalalignment='left', 
        verticalalignment='center'
    );

    fig.savefig(join(figures_path, 'length_bias.pdf'), bbox_inches='tight', dpi=300)
    
    # ----------------------------------------------------------
    # -- Value estimates ---------------------------------------
    # ----------------------------------------------------------

    models = ['mlp', 'pronet_10_2', 'pronet_10_5', 'pronet_100_5', 'mu_true']
    wis_data_ = wis_data[wis_data.model.isin(models)]

    wis_data_15 = wis_data_[wis_data_['time']==15].groupby('model').apply(lambda df : df['v_diff'].unique()).explode()
    estimates = pd.concat([wis_data_15.groupby('model').mean(), wis_data_15.groupby('model').std()], axis=1)
    estimates.rename(columns={0: 'mean', 1: 'std'}, inplace=True)
    print(estimates)
    print()

    wis_data_gt = wis_data_[wis_data_['time']==15] if 'gt' in wis_data_.columns else pd.read_csv(join(config.results.path, 'wis_data_gt.csv'))
    pi_mean = wis_data_gt[wis_data_gt.model=='mlp'].groupby('v_diff').first().reset_index()['gt'].mean()
    pi_std = wis_data_gt[wis_data_gt.model=='mlp'].groupby('v_diff').first().reset_index()['gt'].std()
    print('Average value of target policy: %.2f (%.2f)' % (pi_mean, pi_std))

if __name__ == '__main__':
    args = get_args()
    config_path = args.config_path
    config = load_config(config_path)
    main(config)
