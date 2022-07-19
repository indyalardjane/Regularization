import torch

class SR2optiml12(SR2optim):
    def __init__(self, params, nu1=1e-4, nu2=0.9, g1=1.5, g2=1.25, g3=0.5, lmbda=0.001, sigma=0.75, weight_decay=0.2):
        super().__init__(params, nu1=nu1, nu2=nu2, g1=g1, g2=g2, g3=g3, lmbda=lmbda, sigma=sigma, weight_decay=weight_decay)

    def get_step(self, x, grad, sigma, lmbda):
        p = 54**(1/3)/4 * (2 * lmbda/ sigma)**(2/3)
        phi = torch.arccos((2 * lmbda)/(8 * sigma) * (torch.abs(x.data - grad / sigma)/3)**(-3/2))
        step = torch.where(x.data - grad / sigma > p, 
                           2/3 * torch.abs(x.data - grad / sigma) * (1 + torch.cos(2 * torch.pi /3 - 2/3 * phi )) - x.data, 
                           torch.where(x.data - grad / sigma <  -p,
                           -2/3 * torch.abs(x.data - grad / sigma) * (1 + torch.cos(2 * torch.pi /3 - 2/3 * phi )) - x.data, -x.data))
        return step

class SR2optiml23(SR2optim):
    def __init__(self, params, nu1=1e-4, nu2=0.9, g1=1.5, g2=1.25, g3=0.5, lmbda=0.001, sigma=0.75, weight_decay=0.2):
        super().__init__(params, nu1=nu1, nu2=nu2, g1=g1, g2=g2, g3=g3, lmbda=lmbda, sigma=sigma, weight_decay=weight_decay)

    def get_step(self, x, grad, sigma, lmbda):
        phi = torch.arccosh(27 * (x.data - grad / sigma)**2 /16 * (2* lmbda/sigma)**(-3/2))
        A = 2/np.sqrt(3) * (2* lmbda/sigma)**(1/4) * (torch.cosh(phi/3))**(1/2)
        cond = 2/3 * (3 * (2 * lmbda / sigma)**3)**(1/4)
        step = torch.where(x.data - grad / sigma > cond,
                          (A + torch.sqrt((2 * torch.abs(x.data - grad / sigma))/A - A**2) / 2)**3 - x.data,
                          torch.where(x.data - grad / sigma <  -cond,
                          -(A + torch.sqrt((2 * torch.abs(x.data - grad / sigma))/A - A**2) / 2)**3 - x.data, -x.data))
        return step
