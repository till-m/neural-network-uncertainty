"""adapted from github.com/kimjeyoung/SNGP-BERT-Pytorch"""
import torch
from torch import nn
import math
import copy
from .base import BaseNet
from einops import rearrange, repeat, einsum


class SNBaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        nodes_per_layer = 16
        n_hidden = 2
        self.linear_in = nn.utils.parametrizations.spectral_norm(nn.Linear(1, nodes_per_layer))
        self.relu = nn.ReLU()
        self.linear_out = nn.utils.parametrizations.spectral_norm(nn.Linear(nodes_per_layer, nodes_per_layer))
        hidden = []
        for _ in range(n_hidden-1):
            hidden.append(
                    nn.utils.parametrizations.spectral_norm(nn.Linear(nodes_per_layer, nodes_per_layer))
            )
        
        self.hidden = nn.ModuleList(hidden)


    def forward(self, in_: torch.Tensor):
        x = self.linear_in(in_)
        for layer in self.hidden:
            x = nn.functional.relu(layer(x))
        x = self.linear_out(x)
        return x


def RandomFeatureLinear(i_dim, o_dim, bias=True, require_grad=False):
    m = nn.Linear(i_dim, o_dim, bias)
    # https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/bert_sngp.py
    nn.init.normal_(m.weight, mean=0.0, std=0.05)
    # freeze weight
    m.weight.requires_grad = require_grad
    if bias:
        nn.init.uniform_(m.bias, a=0.0, b=2. * math.pi)
        # freeze bias
        m.bias.requires_grad = require_grad
    m.decay = True
    return m


class RFFGPLayer(nn.Module):
    def __init__(self,
                 n_in,
                 n_out,
                 gp_kernel_scale=1.0,
                 num_inducing=256,
                 layer_norm_eps=1e-12,
                 scale_random_features=True,
                 normalize_input=True,
                 gp_cov_momentum=0.999,
                 gp_cov_ridge_penalty=1e-2):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self.gp_cov_momentum = gp_cov_momentum

        self.gp_input_scale = 1. / math.sqrt(gp_kernel_scale)
        self.gp_feature_scale = math.sqrt(2. / float(num_inducing))
        self.scale_random_features = scale_random_features
        self.normalize_input = normalize_input

        self._gp_input_normalize_layer = torch.nn.LayerNorm(n_in, eps=layer_norm_eps)
        self._gp_output_layers = nn.ModuleList([nn.Linear(num_inducing, 1, bias=False) for _ in range(n_out)])
        self._random_features = nn.ModuleList([RandomFeatureLinear(n_in, num_inducing) for _ in range(n_out)])

        # Laplace Random Feature Covariance
        # Posterior precision matrix for the GP's random feature coefficients.
        self.register_buffer('initial_precision_matrix', self.gp_cov_ridge_penalty * repeat(torch.eye(num_inducing), '... -> n_out ...', n_out=n_out))
        self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.initial_precision_matrix), requires_grad=False)

    def gp_layer(self, gp_inputs, update_cov):
        # Supports lengthscale for custom random feature layer by directly
        # rescaling the input.
        if self.normalize_input:
            gp_inputs = self._gp_input_normalize_layer(gp_inputs)

        gp_features = torch.stack([rf(gp_inputs) for rf in self._random_features])
        # cosine
        gp_features = torch.cos(gp_features)

        if self.scale_random_features:
            gp_features = gp_features * self.gp_input_scale

        # Computes posterior center (i.e., MAP estimate) and variance.
        gp_output = torch.cat([gol(gp_feature) for gol, gp_feature in zip(self._gp_output_layers, gp_features)], dim=-1)

        if update_cov:
            # update precision matrix
            self.update_cov(gp_features)
        return gp_features, gp_output

    def reset_cov(self):
        self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.initial_precision_matrix), requires_grad=False)

    def update_cov(self, gp_features):
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L346
        batch_size = gp_features.size()[1]
        precision_matrix_minibatch = einsum(gp_features, gp_features, 'n_out n_batch n_ind1, n_out n_batch n_ind2 -> n_out n_ind1 n_ind2', )
        # Updates the population-wise precision matrix.
        if self.gp_cov_momentum > 0:
            # Use moving-average updates to accumulate batch-specific precision
            # matrices.
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (
                    self.gp_cov_momentum * self.precision_matrix +
                    (1. - self.gp_cov_momentum) * precision_matrix_minibatch)
        else:
            # Compute exact population-wise covariance without momentum.
            # If use this option, make sure to pass through data only once.
            precision_matrix_new = self.precision_matrix + precision_matrix_minibatch
        #self.precision_matrix.weight = precision_matrix_new
        self.precision_matrix = torch.nn.Parameter(precision_matrix_new, requires_grad=False)

    def compute_predictive_covariance(self, gp_features):
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L403
        # Computes the covariance matrix of the feature coefficient.
        feature_cov_matrix = torch.linalg.inv(self.precision_matrix)
        # numerical stability trick
        feature_cov_matrix = 0.5 * (feature_cov_matrix + torch.transpose(feature_cov_matrix, dim0=-2, dim1=-1))
        # Computes the covariance matrix of the gp prediction.
        cov_feature_product = einsum(feature_cov_matrix, gp_features, 'n_out n_ind1 n_ind2, n_out n_batch n_ind2 -> n_out n_ind1 n_batch') * self.gp_cov_ridge_penalty
        gp_cov_matrix = einsum(gp_features, cov_feature_product, 'n_out n_batch1 n_ind, n_out n_ind n_batch2 -> n_batch1 n_batch2 n_out')
        return gp_cov_matrix
    
    def compute_predictive_variance(self, gp_features):
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L403
        # Computes the variance of the feature coefficient.
        feature_cov_matrix = torch.linalg.inv(self.precision_matrix) # n_out n_ind1 n_ind2
        feature_cov_matrix = 0.5 * (feature_cov_matrix + torch.transpose(feature_cov_matrix, dim0=-2, dim1=-1))
        # Computes the variances of the gp prediction.
        cov_feature_product = einsum(feature_cov_matrix, gp_features, 'n_out n_ind1 n_ind2, n_out n_batch n_ind2 -> n_out n_ind1 n_batch') * self.gp_cov_ridge_penalty
        gp_var_matrix = einsum(gp_features, cov_feature_product, 'n_out n_batch n_ind, n_out n_ind n_batch -> n_batch n_out')
        return gp_var_matrix

    def forward(self, input_, return_gp_cov: bool = False, return_gp_var: bool = False,
                update_cov: bool = None):
        if update_cov is None:
            update_cov = self.training

        gp_features, gp_output = self.gp_layer(input_, update_cov=update_cov)
        #gp_output = gp_output.reshape(input_shape[:-1] + (gp_output.shape[-1],))
        if return_gp_cov:
            gp_cov_matrix = self.compute_predictive_covariance(gp_features)
            return gp_output, gp_cov_matrix
        elif return_gp_var:
            gp_var_matrix = self.compute_predictive_variance(gp_features)
            return gp_output, gp_var_matrix

        return gp_output


class SNGPNet(BaseNet):
    def __init__(self, n_in=16, n_out=1, base_model=SNBaseNet, base_model_kwargs=dict(), gp_layer_kwargs=dict()) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.base_model = base_model(**base_model_kwargs)
        self.gp_output_layer = RFFGPLayer(n_in=n_in, n_out=n_out, **gp_layer_kwargs)

    def forward(self, input_, **kwargs):
        in_shape = input_.shape
        x = self.base_model(input_)
        res = self.gp_output_layer(rearrange(x, '... c -> (...) c'), **kwargs)
        if 'return_gp_var' in kwargs and kwargs['return_gp_var']:
            res1, res2 = res
            res1 = res1.reshape(x.shape[:-1] + (res1.shape[-1],))
            res2 = res2.reshape(x.shape[:-1] + (res1.shape[-1],))
            return res1, res2
        if 'return_gp_cov' in kwargs and kwargs['return_gp_cov']:
            res1, res2 = res
            res1 = res1.reshape(x.shape[:-1] + (res1.shape[-1],))
            res2 = res2.reshape(x.shape[:-1] + x.shape[:-1] + (res1.shape[-1],))
            return res1, res2
        return res.reshape(x.shape[:-1] + (res.shape[-1],))
