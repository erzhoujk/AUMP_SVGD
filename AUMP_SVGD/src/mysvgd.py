import torch
import torch.optim as optim
import torch.autograd as autograd
import autograd.numpy as np
from tqdm import tqdm, trange
from src.kernel import RBF
import argparse
from memory_profiler import profile
import torch.distributions as D
import torch.nn.functional as F

import gc


class etmySVGD():
    """class that will perform svgd via self.update"""
    def __init__(self, kernel, device,distribution):
        self.k = kernel
        self.d = device
        self.distribution = distribution
        
    def del_tensor_ele(self, arr, index):
        index = index.to(torch.int)

        # print(index)
        arr1 = arr[0:index]
        arr2 = arr[index+1:]
        return torch.cat((arr1, arr2), dim=0)
    def divide(self, theta,  k):
        theta = theta - torch.mean(theta, 0)
        list = torch.norm(theta, p=2, dim=0)
        max_list = torch.topk(list, k=k, dim = 0, largest=True, sorted=False)
        support = max_list.indices

        min_list = torch.topk(list, k=theta.shape[1]-k, dim = 0, largest=False, sorted=False)
        blanket = min_list.indices
        

        return blanket, support

    def score(self, X):
        X_cp = X.clone().requires_grad_(True)
        
        log_prob = self.distribution.log_prob(X_cp)
        
        score_func = autograd.grad(log_prob.sum(), X_cp, retain_graph=False)[0]
        score_func1 = score_func.clone().detach()
        
        
        
            
        
        return score_func1

    

    def svgd_kernel(self,  theta_ab):
        """gaussian rbf kernel function"""
        

        X_cp = theta_ab.clone().requires_grad_(True)
        
       
        Y = theta_ab.clone().detach()
        with torch.no_grad():
            self.k.bandwidth(theta_ab, theta_ab)
        K_XX = self.k(X_cp, Y)
        grad_K = -autograd.grad(K_XX.sum(), X_cp, retain_graph=False)[0]
        grad_K = grad_K.clone().detach()
    
        

        return K_XX, grad_K

    

 


    
    def power(self,theta, part, s, lnpgrad):
        theta_ab1 = torch.cat([theta[:,part],theta[:,s].view(theta.shape[0], 1)], 1)
                
                
        lnpgrad_j1 = lnpgrad[:,s]
                # calculating the kernel matrix
        kxy1, repulsion1 = self.svgd_kernel(theta_ab1) 
        
        
        attraction1 = torch.matmul(kxy1, lnpgrad_j1) 
        return  attraction1 + repulsion1[:,-1]

    def update_blanket(self, theta, lnpgrad, k, lr ):
       
                
        blanket , support = self.divide(theta, k)
                       
                
        theta_ab = torch.cat([theta[:,blanket],theta[:,support]], 1)
                
        lnpgrad_blanket = lnpgrad[:,blanket]
                    # calculating the kernel matrix
        kxy, repulsion = self.svgd_kernel(theta_ab) 
            
        
        attraction = torch.matmul(kxy, lnpgrad_blanket) 
                    

        theta[:,blanket] = theta[:,blanket]  + 0.01*lr*(attraction + repulsion[:, 0:blanket.size(0)]) / theta_ab.size(0)

        return theta, blanket, support


    def update_dim(self, theta, k, lr):

        lnpgrad = self.score(theta)
                
        theta, blanket, support = self.update_blanket(theta,  lnpgrad, k, lr )

            
        grad1_theta = theta.clone().detach()
        grad2_theta = theta.clone().detach()


        iterator_dim = range(theta.shape[1])
        s = -1

        for j in iterator_dim:
            s = s + 1
            
            if s in blanket:
                a = torch.where(blanket == s)[0]
    
                blanket = self.del_tensor_ele(blanket, a)
            if s in support:
                a = torch.where(support == s)[0]
    
                support = self.del_tensor_ele(support, a)

        #### STEP2 blanket_to_d
            
            grad1 = self.power(grad1_theta, blanket, s, lnpgrad)

        ### STEP3 SUPPORT_to_d
            grad2 = self.power(theta, support, s, lnpgrad)
            


        
        
            vec1 = 0.02*lr*grad1 / (blanket.shape[0] + 1)
            vec2 = lr*grad2 / (support.shape[0] + 1)
        
            grad1_theta[:,s] =  vec1 
            
        
            grad2_theta[:,s] = vec2 
            
        
        


        return grad1_theta, grad2_theta



    
    





    # @profile
    def update(self, x0, n_iter = 1000, k = 8, lr=0.001):

        
        theta = x0.clone()
        x = x0.clone()

        verbose = True
        iterator = tqdm(range(n_iter)) 
         
        
        

        for  i in iterator:
            a, b = self.update_dim(theta, k, lr)
            theta = theta + a
            theta = theta + b
 
        return theta





class SVGDLR(etmySVGD):
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
                pam, pamrf = self.step(x0, X_batch=X_batch, y_batch=y_batch)
                train_steps = ep * len(train_loader) + j
        
                if train_steps % save_every == 0:
                    self.particles.append((ep, x0.clone().detach()))
                    _, _, test_acc, test_ll = self.p.evaluation(x0.clone().detach(), X_test, y_test)
                    valid_prob, _, valid_acc, valid_ll = self.p.evaluation(x0.clone().detach(), X_valid, y_valid)
                    self.test_accuracy.append((train_steps, test_acc, test_ll))
                    self.valid_accuracy.append((train_steps, valid_acc, valid_ll))

                    if train_steps % 100 == 0:
                        iterator.set_description(f"Epoch {ep} batch {j} accuracy: {valid_acc}, ll: {valid_ll}")