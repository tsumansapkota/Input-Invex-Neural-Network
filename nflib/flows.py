"""
Implements various flows.
Each flow is invertible so it can be forward()ed and inverse()ed.
Notice that inverse() is not backward as in backprop but simply inversion.
Each flow also outputs its log det J "regularization"

Reference:

NICE: Non-linear Independent Components Estimation, Dinh et al. 2014
https://arxiv.org/abs/1410.8516

Variational Inference with Normalizing Flows, Rezende and Mohamed 2015
https://arxiv.org/abs/1505.05770

Density estimation using Real NVP, Dinh et al. May 2016
https://arxiv.org/abs/1605.08803
(Laurent's extension of NICE)

Improved Variational Inference with Inverse Autoregressive Flow, Kingma et al June 2016
https://arxiv.org/abs/1606.04934
(IAF)

Masked Autoregressive Flow for Density Estimation, Papamakarios et al. May 2017 
https://arxiv.org/abs/1705.07057
"The advantage of Real NVP compared to MAF and IAF is that it can both generate data and estimate densities with one forward pass only, whereas MAF would need D passes to generate data and IAF would need D passes to estimate densities."
(MAF)

Glow: Generative Flow with Invertible 1x1 Convolutions, Kingma and Dhariwal, Jul 2018
https://arxiv.org/abs/1807.03039

"Normalizing Flows for Probabilistic Modeling and Inference"
https://arxiv.org/abs/1912.02762
(review paper)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import abc


# ------------------------------------------------------------------------
class Flow(nn.Module):
    """
    Base flow model
    """
    def __init__(self):
        super().__init__()

    def _forward_yes_logDetJ(self, x):
        raise NotImplementedError

    def _forward_no_logDetJ(self, x):
        raise NotImplementedError

    def _inverse_yes_logDetJ(self, z):
        raise NotImplementedError

    def _inverse_no_logDetJ(self, z):
        raise NotImplementedError
    
    def forward(self, x, logDetJ:bool=False):
        return self._forward_yes_logDetJ(x) if logDetJ else self._forward_no_logDetJ(x)
    
    def inverse(self, z, logDetJ:bool=False):
        return self._inverse_yes_logDetJ(z) if logDetJ else self._inverse_no_logDetJ(z)

    # def get_logdetJ(self):
    #     raise NotImplementedError


class SequentialFlow(Flow):
    """ A sequence of Normalizing Flows is a SequentialFlow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def _forward_yes_logDetJ(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m)
        for flow in self.flows:
            x, ld = flow.forward(x, True)
            log_det += ld
        return x, log_det

    def _forward_no_logDetJ(self, x):
        for flow in self.flows:
            x = flow.forward(x, False)
        return x

    def forward_intermediate(self, x, logDetJ=False):
        return self._forward_intermediate_yes_logDetJ(x) if logDetJ else self._forward_intermediate_no_logDetJ(x)
        
    def _forward_intermediate_yes_logDetJ(self, x):
        m, _ = x.shape
        log_det = [torch.zeros(m)]
        zs = [x.data.cpu()]
        for flow in self.flows:
            x, ld = flow.forward(x, True)
            log_det += [ld.data.cpu()]
            zs += [x.data.cpu()]
        return zs, log_det

    def _forward_intermediate_no_logDetJ(self, x):
        zs = [x]
        for flow in self.flows:
            x = flow.forward(x, False)
            zs.append(x.data.cpu())
        return zs

    def _inverse_yes_logDetJ(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z, True)
            log_det += ld
        return z, log_det

    def _inverse_no_logDetJ(self, z):
        for flow in self.flows[::-1]:
            z = flow.inverse(z, False)
        return z

    def _inverse_intermediate_yes_logDetJ(self, z):
        m, _ = z.shape
        log_det = [torch.zeros(m)]
        xs = [z.data.cpu()]
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z, True)
            log_det += [ld]
            xs.append(z.data.cpu())
        return xs, log_det

    def _inverse_intermediate_no_logDetJ(self, z):
        xs = [z.data.cpu()]
        for flow in self.flows[::-1]:
            z = flow.inverse(z, False)
            xs.append(z.data.cpu())
        return xs

    def inverse_intermediate(self, x, logDetJ=False):
        return self._inverse_intermediate_yes_logDetJ(x) if logDetJ else self._inverse_intermediate_no_logDetJ(x)
    

class NormalizingFlow(nn.Module):
    """ A Normalizing Flow Model is a (flow, prior) pair """
    
    def __init__(self, flow_list, prior):
        super().__init__()
        self.flow = SequentialFlow(flow_list)
        self.prior = prior
    
    def forward(self, x, logDetJ=False, intermediate=False):
        if logDetJ:
            if intermediate:
                zs, log_det = self.flow.forward_intermediate(x, True)
                z = zs[-1]
            else:
                zs, log_det = self.flow.forward(x, True)
                z = zs
            prior_logprob = self.prior.log_prob(z).view(x.size(0), -1).sum(1)
            return zs, log_det, prior_logprob
        else:
            if intermediate:
                zs = self.flow.forward_intermediate(x, False)
                z = zs[-1]
            else:
                zs = self.flow.forward(x, False)
                z = zs
            prior_logprob = self.prior.log_prob(z).view(x.size(0), -1).sum(1)
            return z, prior_logprob

    def inverse(self, z, logDetJ=False, intermediate=False):
        if logDetJ:
            if intermediate:
                xs, log_det = self.flow.inverse_intermediate(z, True)
            else:
                xs, log_det = self.flow.inverse(z, True)
            return xs, log_det
        else:
            if intermediate:
                xs = self.flow.inverse_intermediate(z, False)
            else:
                xs = self.flow.inverse(z, False)
            return z
    
    def sample(self, num_samples:int):
        z = self.prior.sample((num_samples,))
        xs = self.flow.inverse(z, False)
        return xs


# ------------------------------------------------------------------------


class AffineConstantFlow(Flow):
    """ 
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """
    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else torch.zeros(1, dim)
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else torch.zeros(1, dim)
        
    def _forward_yes_logDetJ(self, x):
        z = x * torch.exp(self.s) + self.t
        log_det = torch.sum(self.s, dim=1)
        return z, log_det.expand(len(x))

    def _forward_no_logDetJ(self, x):
        z = x * torch.exp(self.s) + self.t
        return z
    
    def _inverse_yes_logDetJ(self, z):
        x = (z - self.t) * torch.exp(-self.s)
        log_det = torch.sum(-self.s, dim=1)
        return x, log_det.expand(len(x))

    def _inverse_no_logDetJ(self, z):
        x = (z - self.t) * torch.exp(-self.s)
        return x

class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output_v0
    is unit gaussian. As described in Glow paper.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False
    
    def _initialize_data(self, x):
        assert self.s is not None and self.t is not None # for now
        self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
        
        self.s.data[torch.isinf(self.s.data)] = 0
        self.s.data[torch.isnan(self.s.data)] = 0
        
        self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
        self.data_dep_init_done = True

    def forward(self, x, logDetJ=False):
        # first batch is used for init
        if not self.data_dep_init_done:
            self._initialize_data(x)    
        return super().forward(x, logDetJ)

class ActNorm2D(ActNorm):
    '''
    Per channel normalization rather than per neuron normalization
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x, logDetJ=False):
        ### (N, dim, h, w)
        x = x.transpose(1,3)
        xs1 = x.shape #### (N, w, h, dim)
        x = x.reshape(-1, xs1[3]) ## now in shape (N*h*w, dim)
        # the batch of data needs to be reshaped
        if logDetJ:
            z, _logdetJ = super().forward(x, True)
            return z.reshape(xs1).transpose(1,3), _logdetJ
        else:
            return super().forward(x, False).reshape(xs1).transpose(1,3)
        
    
    def inverse(self, z, logDetJ=False):
        ### (N, dim, h, w)
        x = z.transpose(1,3)
        xs1 = x.shape #### (N, w, h, dim)
        x = x.reshape(-1, xs1[3]) ## now in shape (N*h*w, dim)
        # the batch of data needs to be reshaped
        if logDetJ:
            z, _logdetJ = super().inverse(x, True)
            return z.reshape(xs1).transpose(1,3).contiguous() , _logdetJ
        else:
            return super().inverse(x, False).reshape(xs1).transpose(1,3).contiguous()


class LinearFlow(Flow):
    def __init__(self, dim, bias=True, identity_init=True):
        super().__init__()
        self.dim = dim

        _l = nn.Linear(dim, dim, bias)
        if identity_init:
            self.weight = nn.Parameter(torch.eye(dim))#[torch.randperm(dim)])
        else:
            UDV = torch.svd(_l.weight.data.t())
            self.weight = nn.Parameter(UDV[0])
            del UDV
        if bias:
            self.bias = nn.Parameter(_l.bias.data)
        else:
            self.bias = None
        del _l

    def _forward_yes_logDetJ(self, x):
        if self.bias is not None:
            y = x + self.bias
        y = y @ self.weight
        return y, self._logdetJ().expand(x.shape[0])

    def _forward_no_logDetJ(self, x):
        if self.bias is not None:
            y = x + self.bias
        y = y @ self.weight
        return y

    def _inverse_yes_logDetJ(self, y):
        x = y @ self.weight.inverse()
        if self.bias is not None:
            x = x - self.bias
        return x, -self._logdetJ().expand(y.shape[0])

    def _inverse_no_logDetJ(self, y):
        x = y @ self.weight.inverse()
        if self.bias is not None:
            x = x - self.bias
        return x

    def _logdetJ(self):
        # return torch.log(torch.abs(torch.det(self.weight))) 
        ### torch.logdet gives nan if negative determinant
        # return self.weight.det().abs().log()
        return self.weight.det().log()


class BatchNorm1DFlow(Flow):
    """
    Batchnorm adapted to normalizing flows
    Scales + Shifts the flow by (learned) constants and exponentially moving average per dimension.
    """

    def __init__(self, dim, eps=1e-5, momentum=0.9):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.scale = nn.Parameter(torch.zeros(1, dim, requires_grad=True))
        self.shift = nn.Parameter(torch.zeros(1, dim, requires_grad=True))

        self.register_buffer('running_mean', None)
        self.register_buffer('running_var', None)

        self.recent_mean = None
        self.recent_var = None

    def _forward_yes_logDetJ(self, x):
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True)
            self.recent_mean, self.recent_var = mean.data, var.data
            if self.running_mean is None:
                self.running_mean = mean.data
                self.running_var = var.data
            else:
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.data
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.data
        else:
            self.recent_mean, self.recent_var = None, None
            mean = self.running_mean
            var = self.running_var

        var = torch.sqrt(var + self.eps)
        x = (x-mean)/var
        x = x * torch.exp(self.scale) + self.shift

        log_det = self.scale.sum(dim=1) - torch.log(var).sum(dim=1)
        return x, log_det.expand(len(x))

    def _forward_no_logDetJ(self, x):
        if self.training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True)
            self.recent_mean, self.recent_var = mean.data, var.data
            if self.running_mean is None:
                self.running_mean = mean.data
                self.running_var = var.data
            else:
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.data
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.data
        else:
            self.recent_mean, self.recent_var = None, None
            mean = self.running_mean
            var = self.running_var

        var = torch.sqrt(var + self.eps)
        x = (x - mean) / var
        x = x * torch.exp(self.scale) + self.shift
        return x

    def _inverse_yes_logDetJ(self, z):
        # assert not self.training, "Cannot compute inverse in training mode"
        z = (z - self.shift) * torch.exp(-self.scale)
        if self.training:
            var = torch.sqrt(self.recent_var + self.eps)
            z = z * var + self.recent_mean
        else:
            var = torch.sqrt(self.running_var + self.eps)
            z = z * var + self.running_mean

        log_det = - torch.sum(self.scale, dim=1) + torch.log(var).sum(dim=1)
        return z, log_det.expand(len(z))

    def _inverse_no_logDetJ(self, z):
        # assert not self.training, "Cannot compute inverse in training mode"
        z = (z - self.shift) * torch.exp(-self.scale)
        if self.training:
            var = torch.sqrt(self.recent_var + self.eps)
            z = z * var + self.recent_mean
        else:
            var = torch.sqrt(self.running_var + self.eps)
            z = z * var + self.running_mean

        return z

class BatchNorm2DFlow(BatchNorm1DFlow):

    def __init__(self, dim, eps=1e-5, momentum=0.9):
        super().__init__(dim, eps, momentum)
        self.scale = nn.Parameter(torch.zeros(1, dim, 1, 1, requires_grad=True))
        self.shift = nn.Parameter(torch.zeros(1, dim, 1, 1, requires_grad=True))
        pass