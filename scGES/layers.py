import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)
from scvi.distributions import NegativeBinomial
from torch.autograd import Variable
from torch.distributions import Poisson
import torch
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)

from typing import Optional

from torch.distributions import Normal


def semi_supervised_loss(loss_function):
    def new_loss_function(y_true, y_pred):
        mask = (y_true != -1).float()
        y_true_pos = F.relu(y_true)
        loss = loss_function(y_pred, y_true_pos)
        masked_loss = loss * mask
        return masked_loss.mean()
    new_func = new_loss_function
    return new_func



def sampling(mu, log_var):
    var = torch.exp(log_var) + 1e-4
    return Normal(mu, var.sqrt()).rsample()

def bce(recon_x, x):
    """Computes BCE loss between reconstructed data and ground truth data.

       Parameters
       ----------
       recon_x: torch.Tensor
            Torch Tensor of reconstructed data
       x: torch.Tensor
            Torch Tensor of ground truth data

       Returns
       -------
       MSE loss value
    """
    bce_loss = torch.nn.functional.binary_cross_entropy(recon_x, (x > 0).float(), reduction='none')
    return bce_loss

def mse(recon_x, x):
    """Computes MSE loss between reconstructed data and ground truth data.

       Parameters
       ----------
       recon_x: torch.Tensor
            Torch Tensor of reconstructed data
       x: torch.Tensor
            Torch Tensor of ground truth data

       Returns
       -------
       MSE loss value
    """
    mse_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='none')
    return mse_loss

def poisson(recon_x, x):
    """Computes Poisson NLL between reconstructed data and ground truth data.

       Parameters
       ----------
       recon_x: torch.Tensor
            Torch Tensor of reconstructed data
       x: torch.Tensor
            Torch Tensor of ground truth data

       Returns
       -------
       MSE loss value
    """
    poisson_loss = -Poisson(recon_x).log_prob(x)
    return poisson_loss


def nb(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps=1e-8):
    """
       This negative binomial function was taken from:
       Title: scvi-tools
       Authors: Romain Lopez <romain_lopez@gmail.com>,
                Adam Gayoso <adamgayoso@berkeley.edu>,
                Galen Xing <gx2113@columbia.edu>
       Date: 16th November 2020
       Code version: 0.8.1
       Availability: https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/distributions/_negative_binomial.py

       Computes negative binomial loss.
       Parameters
       ----------
       x: torch.Tensor
            Torch Tensor of ground truth data.
       mu: torch.Tensor
            Torch Tensor of means of the negative binomial (has to be positive support).
       theta: torch.Tensor
            Torch Tensor of inverse dispersion parameter (has to be positive support).
       eps: Float
            numerical stability constant.

       Returns
       -------
       If 'mean' is 'True' NB loss value gets returned, otherwise Torch tensor of losses gets returned.
    """
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))

    log_theta_mu_eps = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )

    return res

def nb_dist(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, eps=1e-8):
    loss = -NegativeBinomial(mu=mu, theta=theta).log_prob(x)
    return loss


def zinb(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor, pi: torch.Tensor, eps=1e-8):
    """
       This zero-inflated negative binomial function was taken from:
       Title: scvi-tools
       Authors: Romain Lopez <romain_lopez@gmail.com>,
                Adam Gayoso <adamgayoso@berkeley.edu>,
                Galen Xing <gx2113@columbia.edu>
       Date: 16th November 2020
       Code version: 0.8.1
       Availability: https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/distributions/_negative_binomial.py

       Computes zero inflated negative binomial loss.
       Parameters
       ----------
       x: torch.Tensor
            Torch Tensor of ground truth data.
       mu: torch.Tensor
            Torch Tensor of means of the negative binomial (has to be positive support).
       theta: torch.Tensor
            Torch Tensor of inverses dispersion parameter (has to be positive support).
       pi: torch.Tensor
            Torch Tensor of logits of the dropout parameter (real support)
       eps: Float
            numerical stability constant.

       Returns
       -------
       If 'mean' is 'True' ZINB loss value gets returned, otherwise Torch tensor of losses gets returned.
    """
    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    softplus_pi = F.softplus(-pi)  #  uses log(sigmoid(x)) = -softplus(-x)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    return res


def pairwise_distance(x, y):
    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output


def gaussian_kernel_matrix(x, y, alphas):
    """Computes multiscale-RBF kernel between x and y.

       Parameters
       ----------
       x: torch.Tensor
            Tensor with shape [batch_size, z_dim].
       y: torch.Tensor
            Tensor with shape [batch_size, z_dim].
       alphas: Tensor

       Returns
       -------
       Returns the computed multiscale-RBF kernel between x and y.
    """

    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)

    alphas = alphas.view(alphas.shape[0], 1)
    beta = 1. / (2. * alphas)

    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)


def mmd_loss_calc(source_features, target_features):
    """Initializes Maximum Mean Discrepancy(MMD) between source_features and target_features.

       - Gretton, Arthur, et al. "A Kernel Two-Sample Test". 2012.

       Parameters
       ----------
       source_features: torch.Tensor
            Tensor with shape [batch_size, z_dim]
       target_features: torch.Tensor
            Tensor with shape [batch_size, z_dim]

       Returns
       -------
       Returns the computed MMD between x and y.
    """
    alphas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    alphas = Variable(torch.FloatTensor(alphas)).to(device=source_features.device)

    cost = torch.mean(gaussian_kernel_matrix(source_features, source_features, alphas))
    cost += torch.mean(gaussian_kernel_matrix(target_features, target_features, alphas))
    cost -= 2 * torch.mean(gaussian_kernel_matrix(source_features, target_features, alphas))

    return cost


def mmd(y,c,n_conditions, beta, boundary):
    """Initializes Maximum Mean Discrepancy(MMD) between every different condition.

       Parameters
       ----------
       n_conditions: integer
            Number of classes (conditions) the data contain.
       beta: float
            beta coefficient for MMD loss.
       boundary: integer
            If not 'None', mmd loss is only calculated on #new conditions.
       y: torch.Tensor
            Torch Tensor of computed latent data.
       c: torch.Tensor
            Torch Tensor of condition labels.

       Returns
       -------
       Returns MMD loss.
    """

    # partition separates y into num_cls subsets w.r.t. their labels c
    conditions_mmd = partition(y, c, n_conditions)
    loss = torch.tensor(0.0, device=y.device)
    if boundary is not None:
        for i in range(boundary):
            for j in range(boundary, n_conditions):
                if conditions_mmd[i].size(0) < 2 or conditions_mmd[j].size(0) < 2:
                    continue
                loss += mmd_loss_calc(conditions_mmd[i], conditions_mmd[j])
    else:
        for i in range(len(conditions_mmd)):
            if conditions_mmd[i].size(0) < 1:
                continue
            for j in range(i):
                if conditions_mmd[j].size(0) < 1 or i == j:
                    continue
                loss += mmd_loss_calc(conditions_mmd[i], conditions_mmd[j])

    return beta * loss


def one_hot_encoder(idx, n_cls):
    assert torch.max(idx).item() < n_cls
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n_cls)
    onehot = onehot.to(idx.device)
    onehot.scatter_(1, idx.long(), 1)
    return onehot


def partition(data, partitions, num_partitions):
    res = []
    partitions = partitions.flatten()
    for i in range(num_partitions):
        indices = torch.nonzero((partitions == i), as_tuple=False).squeeze(1)
        res += [data[indices]]
    return res


class CondLayers(nn.Module):
    def __init__(
            self,
            n_in: int,
            n_out: int,
            n_cond: int,
            bias: bool,
    ):
        super().__init__()
        self.n_cond = n_cond
        self.expr_L = nn.Linear(n_in, n_out, bias=bias)
        if self.n_cond != 0:
            self.cond_L = nn.Linear(self.n_cond, n_out, bias=False)

    def forward(self, x: torch.Tensor):
        if self.n_cond == 0:
            out = self.expr_L(x)
        else:
            expr, cond = torch.split(x, [x.shape[1] - self.n_cond, self.n_cond], dim=1)
            out = self.expr_L(expr) + self.cond_L(cond)
        return out


class Encoder(nn.Module):

    def __init__(self,
                 layer_sizes: list,
                 use_bn: bool,
                 use_ln: bool,
                 use_dr: bool,
                 dr_rate: float,
                 num_classes: int = 0):
        super().__init__()

        self.n_classes = num_classes

        self.FC = None
        if len(layer_sizes) > 1:
            self.FC = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if i == 0:
                    if num_classes != 0:
                        self.FC.add_module(name="L{:d}".format(i), module=CondLayers(in_size,
                                                                                     out_size,
                                                                                     self.n_classes,
                                                                                     bias=True))
                    else:
                        self.FC.add_module(name="L{:d}".format(i), module=CondLayers(in_size, out_size,
                                                                                     0,  bias=True))

                else:
                    self.FC.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=True))
                if use_bn:
                    self.FC.add_module("N{:d}".format(i), module=nn.BatchNorm1d(out_size, affine=True))
                elif use_ln:
                    self.FC.add_module("N{:d}".format(i), module=nn.LayerNorm(out_size, elementwise_affine=False))
                self.FC.add_module(name="A{:d}".format(i), module=nn.ReLU())
                if use_dr:
                    self.FC.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dr_rate))


    def forward(self, x, batch=None, batch_dim=None):
        if batch is not None:
            batch = one_hot_encoder(batch, n_cls=batch_dim)
            x = torch.cat((x, batch), dim=-1)
        if self.FC is not None:
            x = self.FC(x)

        return x


class Decoder(nn.Module):

    def __init__(self,
                 layer_sizes: list,
                 recon_loss: str,
                 use_bn: bool,
                 use_ln: bool,
                 use_dr: bool,
                 dr_rate: float,
                 num_classes: int = 0):
        super().__init__()
        self.use_dr = use_dr
        self.n_classes = num_classes
        self.recon_loss = recon_loss

        self.FirstL = nn.Sequential()

        if num_classes != 0:
            self.FirstL.add_module(name="L0", module=CondLayers(layer_sizes[0], layer_sizes[1], self.n_classes, bias=False))
        else:
            self.FirstL.add_module(name="L0",module=CondLayers(layer_sizes[0], layer_sizes[1], 0, bias=False))
        if use_bn:
            self.FirstL.add_module("N0", module=nn.BatchNorm1d(layer_sizes[1], affine=True))
        elif use_ln:
            self.FirstL.add_module("N0", module=nn.LayerNorm(layer_sizes[1], elementwise_affine=False))
        self.FirstL.add_module(name="A0", module=nn.ReLU())
        if self.use_dr:
            self.FirstL.add_module(name="D0", module=nn.Dropout(p=dr_rate))

        if len(layer_sizes) > 2:
            self.HiddenL = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[1:-1], layer_sizes[2:])):
                if i+3 < len(layer_sizes):
                    self.HiddenL.add_module(name="L{:d}".format(i+1), module=nn.Linear(in_size, out_size, bias=False))
                    if use_bn:
                        self.HiddenL.add_module("N{:d}".format(i+1), module=nn.BatchNorm1d(out_size, affine=True))
                    elif use_ln:
                        self.HiddenL.add_module("N{:d}".format(i + 1), module=nn.LayerNorm(out_size, elementwise_affine=False))
                    self.HiddenL.add_module(name="A{:d}".format(i+1), module=nn.ReLU())
                    if self.use_dr:
                        self.HiddenL.add_module(name="D{:d}".format(i+1), module=nn.Dropout(p=dr_rate))
        else:
            self.HiddenL = None


        if self.recon_loss == "mse":
            self.recon_decoder = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.ReLU())
        if self.recon_loss == "zinb":
            self.mean_decoder = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.Softmax(dim=-1))
            self.dropout_decoder = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        if self.recon_loss == "nb":
            self.mean_decoder = nn.Sequential(nn.Linear(layer_sizes[-2], layer_sizes[-1]), nn.Softmax(dim=-1))

    def forward(self, z, batch=None, hvg_emb=None, batch_dim=None):
        if batch is not None and hvg_emb is not None:
            batch = one_hot_encoder(batch, n_cls=batch_dim)
            z_cat = torch.cat((hvg_emb, z, batch), dim=-1)
            dec_latent = self.FirstL(z_cat)
        elif batch is not None and hvg_emb is None:
            batch = one_hot_encoder(batch, n_cls=batch_dim)
            z_cat = torch.cat((z, batch), dim=-1)
            dec_latent = self.FirstL(z_cat)
        else:
            dec_latent = self.FirstL(z)


        if self.HiddenL is not None:
            x = self.HiddenL(dec_latent)
        else:
            x = dec_latent


        if self.recon_loss == "mse":
            recon_x = self.recon_decoder(x)
            return recon_x, dec_latent
        elif self.recon_loss == "zinb":
            dec_mean_gamma = self.mean_decoder(x)
            dec_dropout = self.dropout_decoder(x)
            return dec_mean_gamma, dec_dropout, dec_latent
        elif self.recon_loss == "nb":
            dec_mean_gamma = self.mean_decoder(x)
            return dec_mean_gamma, dec_latent

class Classifier(nn.Module):
    def __init__(self,
                 layer_sizes: list,
                 use_bn: bool,
                 use_ln: bool,
                 use_dr: bool,
                 dr_rate: float):
        super().__init__()
        self.use_dr = use_dr

        self.ClassL = nn.Sequential()

        self.ClassL.add_module(name="L0", module=nn.Linear(layer_sizes[0], layer_sizes[1], bias=False))
        if use_bn:
            self.ClassL.add_module("N0", module=nn.BatchNorm1d(layer_sizes[1], affine=True))
        elif use_ln:
            self.ClassL.add_module("N0", module=nn.LayerNorm(layer_sizes[1], elementwise_affine=False))

        if len(layer_sizes) > 2:
            self.ClassL.add_module(name="A0", module=nn.ReLU())
        else:
            self.ClassL.add_module(name="A0", module=nn.Softmax())
        if self.use_dr:
            self.ClassL.add_module(name="D0", module=nn.Dropout(p=dr_rate))


        if len(layer_sizes) > 2:
            self.HiddenL = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[1:-1], layer_sizes[2:])):
                if i + 2 < len(layer_sizes):  # i+3 --> i+2:
                    self.HiddenL.add_module(name="L{:d}".format(i + 1), module=nn.Linear(in_size, out_size, bias=False))
                    if use_bn:
                        self.HiddenL.add_module("N{:d}".format(i + 1), module=nn.BatchNorm1d(out_size, affine=True))
                    elif use_ln:
                        self.HiddenL.add_module("N{:d}".format(i + 1),
                                                module=nn.LayerNorm(out_size, elementwise_affine=False))
                    self.HiddenL.add_module(name="A{:d}".format(i + 1), module=nn.Softmax())
                    if self.use_dr:
                        self.HiddenL.add_module(name="D{:d}".format(i + 1), module=nn.Dropout(p=dr_rate))
        else:
            self.HiddenL = None

    def forward(self, z, hvg_emb = None):

        if hvg_emb is not None:
            z_cat = torch.cat((hvg_emb, z), dim=-1)
            clc_emb = self.ClassL(z_cat)
        else:
            clc_emb = self.ClassL(z)


        if self.HiddenL is not None:
            clc = self.HiddenL(clc_emb)
        else:
            clc = clc_emb

        return clc

class scGES(nn.Module):

    def __init__(self,
                 params: list,
                 dr_rate: float = 0, #0.05
                 dispersion = "gene",
                 recon_loss: Optional[str] = 'nb',
                 n_labels = None,
                 freeze = False,
                 use_bn: bool = True,
                 use_ln: bool = False, # use_ln
                 ):
        super().__init__()
        assert isinstance(params, list)
        assert recon_loss in ["mse", "nb", "zinb", "nb+MSE"], "'recon_loss' must be 'mse', 'nb' or 'zinb'"


        self.input_dim = params[0]
        self.hidden_dim = params[1]
        self.latent_dim = params[2]
        self.class_dim = params[3]
        self.atlas_dim = params[4]
        self.query_dim = params[5]
        self.hvg_dim = params[6]
        self.dispersion = dispersion
        self.recon_loss = recon_loss
        self.freeze = freeze
        self.n_labels = n_labels

        self.use_bn = use_bn
        self.use_ln = use_ln
        self.dr_rate = dr_rate
        if self.dr_rate > 0:
            self.use_dr = True
        else:
            self.use_dr = False

        if self.query_dim != 0:
            self.batch_dim = self.atlas_dim + self.query_dim
        else:
            self.batch_dim = self.atlas_dim

        if self.dispersion == "gene":
            self.theta = torch.nn.Parameter(torch.randn(self.input_dim))
        elif self.dispersion == "gene-batch":
            self.theta = torch.nn.Parameter(torch.randn(self.input_dim, self.batch_dim))
        elif self.dispersion == "gene-label":
            self.theta = torch.nn.Parameter(torch.randn(self.input_dim, self.n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        encoder_layer_sizes = [self.input_dim + self.atlas_dim, self.hidden_dim, self.latent_dim]
        decoder_layer_sizes = [self.latent_dim + self.atlas_dim + self.hvg_dim, self.hidden_dim, self.input_dim]
        class_layer_sizes = [self.latent_dim + self.hvg_dim, self.class_dim]


        self.encoder = Encoder(encoder_layer_sizes,
                               self.use_bn,
                               self.use_ln,
                               self.use_dr,
                               self.dr_rate,
                               self.query_dim)
        self.decoder = Decoder(decoder_layer_sizes,
                               self.recon_loss,
                               self.use_bn,
                               self.use_ln,
                               self.use_dr,
                               self.dr_rate,
                               self.query_dim)
        self.classifier = Classifier(class_layer_sizes,
                                     self.use_bn,
                                     self.use_ln,
                                     self.use_dr,
                                     self.dr_rate)

    def forward(self, x =None, x_c = None, sizefactor=None, batch=None,  hvg_emb =None, minus_num = 1):

        x_log = x_c

        z1 = self.encoder(x, batch, self.batch_dim)

        outputs = self.decoder(z1, batch, hvg_emb, self.batch_dim)

        cls = self.classifier(z1, hvg_emb)

        if self.dispersion == "gene-label":
            theta_ = F.linear(one_hot_encoder(batch, self.n_labels), self.theta)
        elif self.dispersion == "gene-batch":
            theta_ = F.linear(one_hot_encoder(batch, self.batch_dim), self.theta)
        elif self.dispersion == "gene":
            theta_ = self.theta
        theta_ = torch.exp(theta_)

        if self.recon_loss == "mse":
            recon_x, y1 = outputs
            recon_loss = mse(recon_x, x).sum(dim=-1).mean()

        elif self.recon_loss == "poisson":
            dec_mean_gamma, y1 = outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(dec_mean_gamma.size(0), dec_mean_gamma.size(1))
            dec_mean = dec_mean_gamma * size_factor_view
            recon_loss = poisson(dec_mean, x_log).sum(dim=-1).mean()

        elif self.recon_loss == "zinb":
            dec_mean_gamma, dec_dropout, y1 = outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(dec_mean_gamma.size(0), dec_mean_gamma.size(1))
            dec_mean = dec_mean_gamma * size_factor_view
            recon_loss = -zinb(x=x_log, mu=dec_mean, theta=theta_, pi=dec_dropout).sum(dim=-1).mean()

        elif self.recon_loss == "nb":
            dec_mean_gamma, y1 = outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(dec_mean_gamma.size(0), dec_mean_gamma.size(1))
            dec_mean = dec_mean_gamma * size_factor_view


            if batch is None:
                recon_loss = -nb(x=x_log, mu=dec_mean, theta=theta_).sum(dim=-1).mean()

            else:
                ML_batch = nb(x=x_log, mu=dec_mean, theta=theta_).sum(dim=-1)
                abs_value = torch.abs(ML_batch)
                ML_batch_new = ML_batch / minus_num
                recon_loss = -ML_batch_new.mean()

        return recon_loss, z1, cls, dec_mean


class TripletNetworkTNN(nn.Module):
    def __init__(self, base_model):
        super(TripletNetworkTNN, self).__init__()

        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.classifier = base_model.classifier
        self.theta = base_model.theta
        self.dispersion = base_model.dispersion
        self.batch_dim = base_model.batch_dim
        self.recon_loss = base_model.recon_loss


    def forward(self, x = None, x_c = None, sizefactor = None, batch = None, hvg_emb = None, n_labels = None, minus_num = 1):

        x_log = x_c

        if self.dispersion == "gene-label":
            theta_ = F.linear(one_hot_encoder(batch[0], n_labels), self.theta)
        elif self.dispersion == "gene-batch":
            theta_ = F.linear(one_hot_encoder(batch[0], self.batch_dim), self.theta)
        elif self.dispersion == "gene":
            theta_ = self.theta
        theta_ = torch.exp(theta_)


        z0 = self.encoder(x[0], batch[0], self.batch_dim)
        z1 = self.encoder(x[1], batch[1], self.batch_dim)
        z2 = self.encoder(x[2], batch[2], self.batch_dim)


        outputs = self.decoder(z0, batch[0], hvg_emb, self.batch_dim)


        cls = self.classifier(z0, hvg_emb)

        if self.recon_loss == "mse":
            recon_x, y1 = outputs
            recon_loss = mse(recon_x, x_log).sum(dim=-1).mean()

        elif self.recon_loss == "poisson":
            dec_mean_gamma, y1 = outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(dec_mean_gamma.size(0), dec_mean_gamma.size(1))
            dec_mean = dec_mean_gamma * size_factor_view
            recon_loss = poisson(dec_mean, x_log).sum(dim=-1).mean()

        elif self.recon_loss == "zinb":
            dec_mean_gamma, dec_dropout, y1 = outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(dec_mean_gamma.size(0), dec_mean_gamma.size(1))
            dec_mean = dec_mean_gamma * size_factor_view
            recon_loss = -zinb(x=x_log, mu=dec_mean, theta=theta_, pi=dec_dropout).sum(dim=-1).mean()

        elif self.recon_loss == "nb":
            dec_mean_gamma, y1 = outputs
            size_factor_view = sizefactor.unsqueeze(1).expand(dec_mean_gamma.size(0), dec_mean_gamma.size(1))
            dec_mean = dec_mean_gamma * size_factor_view
            recon_loss = -nb(x=x_log, mu=dec_mean, theta=theta_).sum(dim=-1).mean()



        return recon_loss, z0, z1, z2, cls, dec_mean

