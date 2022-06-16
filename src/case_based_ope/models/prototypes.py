import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from case_based_ope.utils.misc import compute_squared_distances
from case_based_ope.utils.data import uses_placeholder_y
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from skorch.dataset import unpack_data
from skorch.utils import to_numpy
from skorch.utils import params_for
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import PackedSequence

class NNEncoder(nn.Module):
    def __init__(self, input_size, output_size=64, hidden_sizes=()):
        super(NNEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.nonlinearity = nn.ReLU()

        sizes = [input_size, *hidden_sizes, output_size]
        self.layers = nn.ModuleList([])
        for i in range(len(sizes)-1):
            in_features = sizes[i]
            out_features = sizes[i+1]
            self.layers += [nn.Linear(in_features, out_features)]

    def forward(self, inputs):
        for layer in self.layers:
            inputs = self.nonlinearity(layer(inputs))
        return inputs

class RNNEncoder(nn.Module):
    def __init__(self, input_size, output_size=64, n_layers=1):
        super(RNNEncoder, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size, output_size, n_layers, batch_first=True)

    def forward(self, inputs):
        '''
        Parameters
        ----------
        inputs : a PackedSequence object
            Input data.
        '''
        encodings, _ = self.rnn(inputs)  # "encodings" is a PackedSequence
        encodings, lengths = pad_packed_sequence(encodings, batch_first=True)  # "encodings" is a (padded) Tensor
        lengths = lengths.cpu().tolist()
        max_length = max(lengths)
        ii = np.concatenate([i*max_length+np.arange(l) for i, l in enumerate(lengths)])  # Ignore padding
        hidden_size = encodings.shape[-1]
        encodings = encodings.view(-1, hidden_size)
        return encodings[ii]

class Prototype(nn.Module):
    def __init__(self, input_size, output_size):
        super(Prototype, self).__init__()
        
        self.input_size = input_size  # hidden_size
        self.output_size = output_size  # n_prototypes
        
        k = 1 / np.sqrt(input_size)
        prototypes = torch.rand(output_size, input_size, requires_grad=True)
        self.prototypes = Parameter(k + ((-k) - k) * prototypes)
    
    def forward(self, inputs):
        squared_distances = compute_squared_distances(inputs, self.prototypes)
        similarities = torch.exp(torch.neg(squared_distances))
        return similarities
    
    def extra_repr(self):
        return 'input_size={}, output_size={}'.format(self.input_size, self.output_size)

class PrototypeNetwork(nn.Module):
    def __init__(self, encoder, n_prototypes, output_size, **kwargs):
        super(PrototypeNetwork, self).__init__()
        
        self.encoder = encoder(**params_for('encoder', kwargs))

        self.hidden_size = self.encoder.output_size
        self.n_prototypes = n_prototypes
        self.output_size = output_size
        
        self.should_use_prototypes = (n_prototypes > 0)

        if self.should_use_prototypes:
            self.prototype_layer = Prototype(self.hidden_size, n_prototypes)
            self.output_layer = nn.Linear(n_prototypes, output_size)
        else:
            self.output_layer = nn.Linear(self.hidden_size, output_size)

    def forward(self, inputs):
        encodings = self.encoder(inputs)
        if self.should_use_prototypes:
            similarities = self.prototype_layer(encodings)
            outputs = self.output_layer(similarities)
        else:
            similarities = torch.Tensor([0])  # Instead of None
            outputs = self.output_layer(encodings)
        return outputs, similarities, encodings

    def set_prototypes(self, new_prototypes):
        if self.should_use_prototypes:
            self.prototype_layer.prototypes.data = new_prototypes
        else:
            pass

class PrototypeClassifier(NeuralNetClassifier):
    def __init__(
        self, 
        d_min=1,
        lambda_div=1e-3,
        lambda_ev=1e-3,
        lambda_cl=1e-3,
        projection_interval=5,
        should_record_losses=True,
        n_prediction_prototypes=None,
        **kwargs
    ):
        super(PrototypeClassifier, self).__init__(**kwargs)

        if projection_interval > 0:
            assert self.max_epochs % projection_interval == 0

        if n_prediction_prototypes is not None:
            warnings.warn('Was it intended to set the number of prediction prototypes already?')

        self.d_min = d_min
        self.lambda_div = lambda_div
        self.lambda_ev = lambda_ev
        self.lambda_cl = lambda_cl

        self.projection_interval = projection_interval

        self.should_record_losses = should_record_losses

        self.n_prediction_prototypes = None

    @property
    def _default_callbacks(self):
        return [
            ('train_acc', EpochScoring(scoring='accuracy', lower_is_better=False, on_train=True, name='train_acc')),
            ('valid_acc', EpochScoring(scoring='accuracy', lower_is_better=False, name='valid_acc'))
        ]

    def on_batch_begin(self, net, Xi=None, yi=None, training=False, **kwargs):
        pass
    
    def on_batch_end(self, net, Xi=None, yi=None, training=False, **kwargs):
        pass

    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        self.history.new_epoch()
        self.history[-1].pop('batches')  # Do not store any batch information
        self.history.record('epoch', len(self.history))

    def project_prototypes(self, net, iterator):
        inputs = []
        similarities = []
        encodings = []
        for data in iterator:
            Xi, _ = unpack_data(data)
            with torch.no_grad():
                net.module_.eval()
                _, batch_similarities, batch_encodings = net.infer(Xi)
            if isinstance(Xi, PackedSequence):
                Xi, lengths = pad_packed_sequence(Xi, batch_first=True)
                lengths = lengths.cpu().tolist()
                max_length = max(lengths)
                ii = np.concatenate([j*max_length+np.arange(l) for j, l in enumerate(lengths)])
                input_size = Xi.shape[-1]
                Xi = Xi.view(-1, input_size)
                inputs += [Xi[ii]]
            else:
                inputs += [Xi]
            similarities += [batch_similarities]
            encodings += [batch_encodings]
        inputs = torch.cat(inputs, dim=0);
        similarities = torch.cat(similarities, dim=0)
        encodings = torch.cat(encodings, dim=0)
        _, max_indices = torch.max(similarities, dim=0)
        projections = encodings[max_indices]
        input_prototypes = inputs[max_indices].cpu().numpy()
        max_indices = max_indices.cpu().numpy()
        return projections, input_prototypes, max_indices

    def _get_avg_losses(self, net, iterator):
        '''
        This function is only called during validation.
        '''
        avg_losses = {}
        for i, data in enumerate(iterator):
            Xi, yi = unpack_data(data)
            with torch.no_grad():
                net.module_.eval()
                yp = net.infer(Xi)
                loss_terms = self._get_loss_terms(yp, yi, X=Xi, training=False)
            if i == 0:
                for loss_term in loss_terms.keys():
                    avg_losses[loss_term] = 0
            for loss_term, loss in loss_terms.items():
                avg_losses[loss_term] += loss.item()
        
        n = i+1
        for loss_term in avg_losses.keys():
            avg_losses[loss_term] *= 1/n
        
        return avg_losses

    def record_losses(self, net, iterator, training):
        prefix = 'train_' if training else 'valid_'
        avg_losses = self._get_avg_losses(net, iterator)
        tot_loss = 0
        for loss_term, loss in avg_losses.items():
            tot_loss += loss
            net.history.record(prefix+loss_term, loss)
        net.history.record(prefix+'tot_loss', tot_loss)

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        iterator_train = net.get_iterator(dataset_train, training=True)
        
        n_completed_epochs = net.history[-1]['epoch']
        if self.projection_interval > 0 and n_completed_epochs % self.projection_interval == 0:
            assert hasattr(self.module_, 'prototype_layer')
            new_prototypes, input_prototypes, indices = self.project_prototypes(net, iterator_train)
            if not self.train_split:
                # The indices make only sense if train_split=False
                net.history.record('prototype_indices', indices)
            net.history.record('prototypes', input_prototypes)
            new_prototypes = new_prototypes.to(self.device)
            self.module_.set_prototypes(new_prototypes)

        if self.should_record_losses:
            self.record_losses(net, iterator_train, training=True)

        if dataset_valid:
            iterator_valid = net.get_iterator(dataset_valid, training=False)

            if self.should_record_losses:
                self.record_losses(net, iterator_valid, training=False)

    def run_single_epoch(self, dataset, training, prefix, step_fn, **fit_params):
        is_placeholder_y = uses_placeholder_y(dataset)
        batch_count = 0
        for data in self.get_iterator(dataset, training=training):
            Xi, yi = unpack_data(data)
            yi_res = yi if not is_placeholder_y else None
            self.notify('on_batch_begin', X=Xi, y=yi_res, training=training)
            step = step_fn(Xi, yi, **fit_params)
            self.notify('on_batch_end', X=Xi, y=yi_res, training=training, **step)
            batch_count += 1
        self.history.record(prefix + '_batch_count', batch_count)

    def _get_loss_terms(self, y_pred, y_true, X, training):
        loss_terms = {}
        
        outputs, _, encodings = y_pred
        
        # Cross-entropy
        if self.module_.output_size == 1:
            y_true = y_true.view(outputs.size()).type_as(outputs)
        ce_loss = super(PrototypeClassifier, self).get_loss(outputs, y_true, X, training)
        loss_terms['ce_loss'] = ce_loss

        if hasattr(self.module_, 'prototype_layer'):
            prototypes = self.module_.prototype_layer.prototypes
            if not training:
                prototypes = prototypes.detach()
                encodings = encodings.detach()
            pdistances = F.pdist(prototypes).unsqueeze(0)
            sq_distances = compute_squared_distances(encodings, prototypes)

            # Diversity
            zeros = torch.zeros(pdistances.shape).to(self.device)
            temp = torch.cat([zeros, self.d_min-pdistances], dim=0)
            div_reg = torch.square(torch.max(temp, dim=0)[0]).sum()
            loss_terms['div_loss'] = self.lambda_div*div_reg

            # Clustering
            cl_reg = sq_distances.min(dim=1)[0].sum()
            loss_terms['cl_loss'] = self.lambda_cl*cl_reg

            # Evidence
            ev_reg = sq_distances.min(dim=0)[0].sum()
            loss_terms['ev_loss'] = self.lambda_ev*ev_reg
        
        return loss_terms

    def get_loss(self, y_pred, y_true, X=None, training=False):
        loss_terms = self._get_loss_terms(y_pred, y_true, X, training)
        tot_loss = 0
        for _, loss_term in loss_terms.items():
            tot_loss += loss_term
        return tot_loss

    def make_predictions(self, X):
        assert hasattr(self.module_, 'prototype_layer')
        
        with torch.no_grad():
            self.module_.eval()
            _, similarities, _ = self.infer(X)
        
        similarities_ = torch.zeros(similarities.size()).to(self.device)
        ii = torch.argsort(torch.argsort(similarities, dim=-1), dim=-1)
        N = self.module_.n_prototypes - self.n_prediction_prototypes
        similarities_.flatten()[ii.flatten() >= N] = similarities.flatten()[ii.flatten() >= N]

        nonlinearity = self._get_predict_nonlinearity()
        with torch.no_grad():
            self.module_.eval()
            yp = nonlinearity(self.module_.output_layer(similarities_))

        return yp

    def predict_proba(self, X):
        if self.n_prediction_prototypes is None:
            return super(PrototypeClassifier, self).predict_proba(X)
        else:
            y_probas = []
            dataset = self.get_dataset(X)
            for data in self.get_iterator(dataset, training=False):
                Xi = unpack_data(data)[0]
                yp = self.make_predictions(Xi)
                y_probas.append(to_numpy(yp))
            y_probas = np.concatenate(y_probas, 0)
            return y_probas

    def predict(self, X):
        yp = self.predict_proba(X)
        return yp.argmax(axis=1)

    def score(self, X, y):
        yp = self.predict(X)
        return accuracy_score(y, yp)
