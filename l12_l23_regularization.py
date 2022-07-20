import torch


#l1/2 regularization
class SR2optiml12(SR2optim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_step(self, x, grad, sigma, lmbda):
        X = x.data - grad / sigma
        p = 54**(1/3)/4 * (2 * lmbda/ sigma)**(2/3)
        a = torch.abs(X)
        phi = torch.arccos(lmbda/(4 * sigma) * (a/3)**(-3/2))
        s = 2/3 * a * (1 + torch.cos(2 * torch.pi /3 - 2/3 * phi ))
        
        step = torch.where(X > p, s - x.data, 
                           torch.where(X <  -p, -s - x.data, -x.data))
        return step


#l2/3 regularization
class SR2optiml23(SR2optim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_step(self, x, grad, sigma, lmbda):
#         logging.basicConfig(level=logging.DEBUG)
#         logging.debug('lambda = {}, sigma = '.format(lmbda, sigma))
        X = x.data - grad / sigma
        L = 2* lmbda/sigma
#         logging.debug('arg arccosh = '.format(torch.max(27 * (x.data - grad / sigma)**2 /16 * (2* lmbda/sigma)**(-3/2))))
#         logging.debug('lambda/sigma puissance ='.format((L)**(-3/2)))
        phi = torch.arccosh(27/16 * X**2 * L**(-3/2))
#         logging.debug('phi = '.format(phi))
        A = 2/np.sqrt(3) * L**(1/4) * (torch.cosh(phi/3))**(1/2)
        cond = 2/3 * (3 * L**3)**(1/4)
        s = ((A + torch.sqrt((2 * torch.abs(X))/A - A**2)) / 2)**3
        
        step = torch.where(X > cond, s - x.data,
                           torch.where(X <  -cond, -s - x.data, -x.data))
        return step
