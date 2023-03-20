import torch
import torch.optim as optim
import torch.autograd as autograd
import autograd.numpy as np
from tqdm import tqdm, trange
from src.kernel import RBF
import argparse
from numpy import linalg

from tqdm import trange
import numpy as np             
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal, gaussian_kde, norm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.distributions as D
import torch.nn.functional as F

def divide(theta,  k):
    
    theta = theta - np.mean(theta, 0)
    list = np.zeros((1,np.shape(theta)[1] ))
    for i  in range(np.shape(theta)[1]):
        list[0,i] = np.linalg.norm(theta[:,i])

    arglist = list.argsort()
    
    blanket = arglist[0, k:]
    support = arglist[0,0:k]
    
def l2norm(X, Y):
    """Compute \|X - Y\|_2^2 of tensors X, Y
    """
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())

    dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)
    return dnorm2

def median_heuristic(dnorm2, device):
    """Compute median heuristic.
    Inputs:
        dnorm2: (n x n) tensor of \|X - Y\|_2^2
    Return:
        med(\|X_i - Y_j\|_2^2, 1 \leq i < j \leq n)
    """
    ind_array = torch.triu(torch.ones_like(dnorm2, device=device), diagonal=1) == 1
    med_heuristic = torch.median(dnorm2[ind_array])
    #med_heuristic = torch.Tensor(1).to('cuda')
    return med_heuristic

        
    
    
    
    return blanket, support

class mySVGD():
    """class that will perform svgd via self.update"""
    def __init__(self,device, distribution):
        self.device = device
        self.distribution = distribution


        
    def del_tensor_ele(self, arr, index):
        index = index.to(torch.int)
        if min(index.shape) == 0:
            return arr
        #print(index)
        arr1 = arr[0:index]
        arr2 = arr[index+1:]
        return torch.cat((arr1, arr2), dim=0)
    def divide(self, theta,  k):
        theta = theta - torch.mean(theta, 0)
        list = torch.norm(theta, p=2, dim=0)
        max_list = torch.topk(list, k=k, dim = 0, largest=True, sorted=False)
        blanket = max_list.indices

        min_list = torch.topk(list, k=theta.shape[1]-k, dim = 0, largest=False, sorted=False)
        support = min_list.indices

        return blanket, support

    
       


    # @jit(nopython=True)
    def svgd_kernel(self, theta,theta_ab):
        """gaussian rbf kernel function"""
        
        sq_dist = F.pdist(theta_ab)

        pairwise_dists = torch.tensor(squareform(sq_dist.cpu().detach().numpy())**2).to(self.device)
        
        h = pairwise_dists.median()
        b = torch.tensor(theta_ab.shape[0], device=self.device)
            
        h = torch.sqrt(0.5 * h / torch.log(b))  
            
            
            
           # h = np.sqrt( h**2 ) 
        # compute the rbf kernel
        Kxy = torch.exp( -pairwise_dists / h**2/2)
        
        
        dxkxy = -torch.matmul(Kxy, theta)
        sumkxy = torch.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + torch.multiply(theta[:,i], sumkxy)
        dxkxy = dxkxy / (h**2)
        return (Kxy, dxkxy)

    
       

    


    def step_update(self,  theta, theta_ab, **kwargs):
        historical_grad = 0
        
        
        
        alpha = 0.9
        fudge_factor = 1e-6
        lnpgrad = self.score(theta,**kwargs)
        # calculating the kernel matrix
        kxy, dxkxy = self.svgd_kernel(theta, theta_ab)  
        grad_theta = (torch.matmul(kxy, lnpgrad) + dxkxy) / theta.shape[0] 
        if iter == 0:
                historical_grad = historical_grad + grad_theta ** 2
        else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
       
        
            
        adj_grad = torch.div(grad_theta, fudge_factor + torch.sqrt(historical_grad))

       
        return adj_grad

    def score(self, X, **kwargs):
        X_cp = X.clone().detach().requires_grad_()
        log_prob = self.distribution.log_prob(X_cp, **kwargs)
        score_func = autograd.grad(log_prob.sum(), X_cp)[0]
        return score_func

    
    



    def update(self, x0,  n_iter = 100, k = 8, debug = False,lr=0.001,  **kwargs):

        if x0 is None or self.distribution is None:
            raise ValueError('x0 or lnprob cannot be None!')
        
        theta = x0.clone()
        theta = theta.to(self.device) 
        
        
        
       
        for iter in trange(n_iter):
            
            
            if debug and (iter+1) % 1000 == 0:
                print ('iter ' + str(iter+1) )
            dim = theta.shape[1]
            s = -1
            
            
                
        
            blanket , support = self.divide(theta=theta, k=k)

            ### STEP1 support_to_blanket
                
            theta_ab = torch.cat([theta[:,blanket],theta[:,support]], 1)
               
            support_to_blanket = self.step_update( theta, theta_ab,**kwargs)
                

            theta[:,blanket] = theta[:,blanket] + 0.001*support_to_blanket[:,blanket]

            #### STEP2 blanket_to_d
            for j in range(dim):
                s = s + 1

                if s in blanket:
                    a = torch.where(blanket == s)[0]
    
                    blanket = self.del_tensor_ele(blanket, a)
                if s in support:
                    a = torch.where(support == s)[0]
    
                    support = self.del_tensor_ele(support, a)

                
                theta_ab = torch.cat([theta[:,blanket],theta[:,s].view(theta.shape[0], 1)], 1)
                blanket_to_d = self.step_update(theta, theta_ab,**kwargs)
                
                

            ### STEP3 SUPPORT_to_d
                
                theta_ab = torch.cat([theta[:,support],theta[:,s].view(theta.shape[0], 1)], 1)
                support_to_d = self.step_update(theta, theta_ab,**kwargs)
                
                support_to_d = 0.1*lr*support_to_d

                
                  
                theta[:,s] = theta[:,s] + support_to_d[:,s] 
                
                 
                theta[:,s] = theta[:,s] + lr*blanket_to_d[:,s]
                
                
                
                


        return theta






class mySVGDLR(mySVGD):
    def fit(self, x0: torch.Tensor, epochs: torch.int64, verbose: bool = True,
        metric: callable = None,
        save_every: int = 100,
        train_loader = None,
        valid_data = None,
        test_data = None
    ):
        """
        Args:
            x0 (torch.Tensor): Initial set of particles to be updated
            epochs (torch.int64): Number of gradient descent iterations
        """
        self.particles = [x0.clone().detach().cpu()]
        self.pam = [0] * (epochs//save_every)
        self.test_accuracy = []
        self.valid_accuracy = []

        X_valid, y_valid = valid_data
        X_test, y_test = test_data

        iterator = trange(epochs) if verbose else range(epochs)

        for ep in iterator:
            for j, (X_batch, y_batch) in enumerate(train_loader):
                theta, _ = self.update(x0,  n_iter = 10, k = 2, debug = False,lr=0.01, vector = 0, X_batch=X_batch, y_batch=y_batch)
                train_steps = ep * len(train_loader) + j
        
                if train_steps % save_every == 0:
                    self.particles.append((ep, theta.clone().detach()))
                    _, _, test_acc, test_ll = self.distribution.evaluation(theta.clone().detach(), X_test, y_test)
                    valid_prob, _, valid_acc, valid_ll = self.distribution.evaluation(theta.clone().detach(), X_valid, y_valid)
                    self.test_accuracy.append((train_steps, test_acc, test_ll))
                    self.valid_accuracy.append((train_steps, valid_acc, valid_ll))
                    print(test_acc)

                    if train_steps % 100 == 0:
                        iterator.set_description(f"Epoch {ep} batch {j} accuracy: {valid_acc}, ll: {valid_ll}")