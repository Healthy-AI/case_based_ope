import pickle
import torch
import yaml
import argparse
import numpy as np
from pathlib import Path
from addict import Dict as Adict

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-c', '--config_path', type=Path, help='Path to config.', required=True)
    return parser.parse_args()

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return Adict(config)

def save_data(data, datapath, filename):
    Path(datapath).mkdir(parents=True, exist_ok=True)
    with open(datapath + filename + '.pickle', 'wb') as f:
        pickle.dump(data, f)

def compute_squared_distances(x1, x2):
    '''Compute squared distances using quadratic expansion.
    
    Reference: https://github.com/pytorch/pytorch/pull/25799.
    '''
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment
    x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x2_pad = torch.ones_like(x2_norm)
    
    x1 = torch.cat([-2. * x1, x1_norm, x1_pad], dim=-1)
    x2 = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    
    return x1.matmul(x2.transpose(-2, -1))

def sigmoid_then_2d(x):
    probas = torch.sigmoid(x).flatten()
    return torch.stack((1 - probas, probas), 1)

def similarity(a, b):
    return np.exp(-np.linalg.norm(a-b)**2)

def compute_similarities(X, x):
    '''
    Parameters
    ----------
    X : NumPy array (n_samples, n_features)
    x : NumPy array (n_features,)
    '''
    return np.apply_along_axis(similarity, 1, X, x)

def get_net_params(default, specified):
    params = {}
    for param, value in default.items():
        if param in (
            'module',
            'criterion',
            'optimizer',
            'iterator_train',
            'iterator_valid',
            'dataset'
        ):
            if param in specified:
                params.update(
                    _get_object_and_parameters(param, value, specified[param])
                )
            else:
                params.update(
                    _get_object_and_parameters(param, value)
                )
        else:
            params[param] = _check_value(specified[param]) if param in specified else _check_value(value)
    return params
