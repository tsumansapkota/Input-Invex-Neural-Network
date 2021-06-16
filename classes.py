import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_children(module):
    child = list(module.children())
    if len(child) == 0:
        return [module]
    children = []
    for ch in child:
        grand_ch = get_children(ch)
        children+=grand_ch
    return children

class LogisticRegression(nn.Module):
    
    def __init__(self, dim=784):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.sigmoid(self.linear(x))
        return x



class Bandpass(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        ##mean, var, a
        self.mean = nn.Parameter(torch.randn(1,input_dim, dtype=torch.float32))
        self.icov = nn.Parameter(torch.eye(input_dim, input_dim, dtype=torch.float32)*2)
        self.a = nn.Parameter(torch.tensor([2], dtype=torch.float32))
        
    def forward(self, x):
        self.a.data = torch.clamp(self.a.data, 0.01, 100)
#         self.a.data = torch.clamp(self.a.data, -100, 100)
        x = x-self.mean
        xm = torch.matmul(x.unsqueeze(1), self.icov)
        xm = torch.matmul(xm, x.unsqueeze(2)).squeeze(1)
        xm = torch.abs(xm)
        xm = torch.exp(-(xm**self.a))
        return xm




class ConvexNN(nn.Module):
    
    def __init__(self, dims:list, actf=nn.LeakyReLU):
        super().__init__()
        assert len(dims)>1
        self.dims = dims
        layers = []
        skip_layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i>0:
                skip_layers.append(nn.Linear(dims[0], dims[i+1]))
                skip_layers[-1].weight.data *= 0.1
                layers[-1].weight.data *= 0.1
            if i<len(dims)-2:
                layers.append(actf())
            
        self.layers = nn.ModuleList(layers)
        self.skip_layers = nn.ModuleList(skip_layers)
        
    def forward(self,x):
        h = x
        for i in range(len(self.dims)-1):
            self.layers[i*2].weight.data = torch.abs(self.layers[i*2].weight.data)
            h = self.layers[i*2](h)
            if i>0:
                h += self.skip_layers[i-1](x)
            if i<len(self.dims)-2:
                h = self.layers[i*2+1](h)
        return h


class BasicInvexNet(nn.Module):
    
    def __init__(self, input_dim, net, lamda=1.):
        super().__init__()
        
        self.net = net
        self.center = nn.Parameter(torch.zeros(input_dim).unsqueeze(0))
        self.lamda = lamda
        self.X = None
        
        self.Y = None
        self.dydx_cent = None
        self.dydx = None
        
        self.gp = 0
        self.gclipper = 999
        
    def forward(self, x, hook=True):
#         self.weight_norm()
        self.X = x
        if not x.requires_grad:
            self.X = torch.autograd.Variable(x, requires_grad=True)
        
        self.Y = self.net(self.X)
        y = self.Y+0.
        if y.requires_grad and hook:
            y.register_hook(self.scale_gradient_back)
        return y
    
    def scale_gradient_back(self, grad):
        return torch.minimum(torch.maximum(grad, -self.gclipper), self.gclipper)
    
    def weight_norm(self):
        for child in get_children(self.net):
            if isinstance(child, nn.Linear):
                norm = torch.norm(child.weight.data, dim=1)
                indx = torch.nonzero(norm>5., as_tuple=False).t()[0]
                if len(indx)==0: continue
                child.weight.data[indx] *= 3/(norm[indx].reshape(-1,1))
        
    def get_dydx_cent(self):
        self.dydx_cent = self.center-self.X.data
        self.dydx_cent = self.dydx_cent/torch.norm(self.dydx_cent, p=2, dim=1, keepdim=True)
        return self.dydx_cent
    
    def get_dydx(self):
        self.dydx = torch.autograd.grad(outputs=self.Y, inputs=self.X,
                                    grad_outputs=torch.ones_like(self.Y),
                                    only_inputs=True, retain_graph=True, create_graph=True)[0]
        return self.dydx
    
    def smooth_l1(self, x, beta=1):
        mask = x<beta
        y = torch.empty_like(x)
        y[mask] = 0.5*(x[mask]**2)/beta
        y[~mask] = torch.abs(x[~mask])-0.5*beta
        return y
    
    def get_gradient_penalty(self):
        m = self.dydx.shape[0]
        ## gradient is projected in the direction of center (aka. minima/maxima)
        projected_grad = torch.bmm(self.dydx.view(m, 1, -1), self.dydx_cent.view(m, -1, 1)).view(-1, 1)
        self.cond = projected_grad
        
        intolerables = F.softplus(self.cond-0.1, beta=-20)
        self.gp = (self.smooth_l1(intolerables*5)).mean()*self.lamda
        return self.gp
    
    def get_gradient_clipper(self):
        with torch.no_grad():
            cond = self.cond.data
            linear_mask = cond>0.14845
            a = 20.
            gclipper = -((1.05*(cond-1))**4)+1
            gclipper = torch.log(torch.exp(a*gclipper)+1)/a
            gc2 = 3*cond-0.0844560006
            gclipper[linear_mask] = gc2[linear_mask]
            self.gclipper = gclipper
        return self.gclipper

    def compute_penalty_and_clipper(self):
        self.get_dydx_cent()
        self.get_dydx()
        self.get_gradient_penalty()
        self.get_gradient_clipper()
        return


class LipschitzInvexNet(nn.Module):
    
    def __init__(self, net_invx, net_lips, lamda=1.):
        super().__init__()
        
        self.net_lips = net_lips
        self.net_invx = net_invx
        self.lamda = lamda
        
        self.X = None
        self.Y_lips = None
        self.Y_invx = None
        self.Y = None

        self.dydx_invx = None
        self.dydx_lips = None
        self.dydx = None
        
        self.gp = 0
        self.gclipper = 999
        
        self._temp_hook_invx = False
        
    def forward(self, x, hook_invx=False, hook_lips=True):
#         self.weight_norm()
        self.X = x
        if not self.X.requires_grad:
            self.X = torch.autograd.Variable(x, requires_grad=True)
                    
        self.Y_invx = self.net_invx(self.X, hook_invx)
        self._temp_hook_invx = hook_invx
    
        self.Y_lips = self.net_lips(self.X)
        y = self.Y_lips+0.
        if hook_lips:
            y.register_hook(self.scale_gradient_back)

        self.Y = self.Y_invx+y
        return self.Y
        
    def get_dydx_invx(self):
        self.dydx_invx = self.net_invx.dydx.data
        return self.dydx_invx
    
    def get_dydx_lips(self):
        self.dydx_lips = torch.autograd.grad(outputs=self.Y_lips, inputs=self.X,
                                    grad_outputs=torch.ones_like(self.Y_lips),
                                    only_inputs=True, retain_graph=True, create_graph=True)[0]
        return self.dydx_lips
    
    def scale_gradient_back(self, grad):
        if self._temp_hook_invx:
            violation = self.net_invx.cond <= 0
            self.gclipper[violation] = torch.zeros_like(self.gclipper)[violation]
        return torch.minimum(torch.maximum(grad, -self.gclipper), self.gclipper)
    
    def weight_norm(self):
        for child in get_children(self.net_lips):
            if isinstance(child, nn.Linear):
                norm = torch.norm(child.weight.data, dim=1)
                indx = torch.nonzero(norm>5., as_tuple=False).t()[0]
                if len(indx)==0: continue
                child.weight.data[indx] *= 3/(norm[indx].reshape(-1,1))
    
    def smooth_l1(self, x, beta=1):
        mask = x<beta
        y = torch.empty_like(x)
        y[mask] = 0.5*(x[mask]**2)/beta
        y[~mask] = torch.abs(x[~mask])-0.5*beta
        return y
    
    def get_gradient_penalty(self):
        norm_dydx_invx = torch.sqrt((self.dydx_invx**2).sum(dim=1, keepdim=True))
        m = self.dydx_lips.shape[0]
        dots = torch.bmm(self.dydx_lips.view(m, 1, -1), self.dydx_invx.view(m, -1, 1)).view(-1, 1)
        norm_proj = dots/norm_dydx_invx
        self.cond = (norm_proj+norm_dydx_invx)

#         intolerables = torch.clamp(F.softplus(self.cond-0.1, beta=-20), -1, 1)
        intolerables = F.softplus(self.cond-0.1, beta=-20)
        self.gp = (self.smooth_l1(intolerables*5)).mean()*self.lamda
        return self.gp
    
    def get_gradient_clipper(self):
        with torch.no_grad():
            cond = self.cond.data
            linear_mask = cond>0.14845
            a = 20.
            gclipper = -((1.05*(cond-1))**4)+1
            gclipper = torch.log(torch.exp(a*gclipper)+1)/a
            gc2 = 3*cond-0.0844560006
            gclipper[linear_mask] = gc2[linear_mask]
            self.gclipper = gclipper
        return self.gclipper

    def compute_penalty_and_clipper(self):
        self.get_dydx_lips()
        self.get_dydx_invx()
        self.dydx = self.dydx_invx+self.dydx_lips
        
        self.get_gradient_penalty()
        self.get_gradient_clipper()
        return