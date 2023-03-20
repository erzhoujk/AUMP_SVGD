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
from scipy.spatial.distance import pdist, squareform
import gc


class mp_gauss_SVGD():
    """class that will perform svgd via self.update"""
    def __init__(self, kernel, device):
        self.k = kernel
        self.d = device

    

    def svgd_kernel(self,  theta_ab):
        """gaussian rbf kernel function"""

        X_cp = theta_ab.clone().detach().requires_grad_()
       
        Y = theta_ab.clone().detach()
        with torch.no_grad():
            self.k.bandwidth(theta_ab, theta_ab)
        K_XX = self.k(X_cp, Y)
        grad_K = -autograd.grad(K_XX.sum(), X_cp)[0] 
        

        return K_XX, grad_K

    

 


    
    def power(self,theta, s, lnpgrad):
        
                
                
        lnpgrad_j1 = lnpgrad[:,s].view(theta.shape[0], 1)
                # calculating the kernel matrix
        kxy, repulsion = self.svgd_kernel(theta[:, s].view(theta.shape[0], 1)) 
        
        
        
        attraction1 = torch.matmul(kxy, lnpgrad_j1) 
        

        return  attraction1 + repulsion




    def update_dim(self,score, theta,  lr):

        lnpgrad = score(theta)
                
        

            
                

        iterator_dim = range(theta.shape[1])
        s = -1

        for j in iterator_dim:
            s = s + 1
  
            
        #### STEP2 blanket_to_d
            
            grad1 = self.power(theta,  s, lnpgrad)
            


        
            vec1 = lr*grad1
            
            
            
        
            theta[:,s] = theta[:,s] + vec1.T

        return theta




    # @profile
    def update(self, x0,score, n_iter = 1000,  lr=0.001):

        
        theta = x0.clone()
       
        iterator = tqdm(range(n_iter)) 
         
        
        

        for  i in iterator:
            theta= self.update_dim(score, theta,  lr)
 
        return theta






class mp_shape_SVGD():
    """class that will perform svgd via self.update"""
    def __init__(self, kernel, device):
        self.k = kernel
        self.d = device

    

    def svgd_kernel(self,  theta_ab):
        """gaussian rbf kernel function"""

        X_cp = theta_ab.clone().detach().requires_grad_()
       
        Y = theta_ab.clone().detach()
        with torch.no_grad():
            self.k.bandwidth(theta_ab, theta_ab)
        K_XX = self.k(X_cp, Y)
        grad_K = -autograd.grad(K_XX.sum(), X_cp)[0] 
        

        return K_XX, grad_K

    

 


    
    def power(self,theta, s, lnpgrad):
        
                
                
        lnpgrad_j1 = lnpgrad[:,s].view(theta.shape[0], 1)
                # calculating the kernel matrix
        kxy, repulsion = self.svgd_kernel(theta[:, s].view(theta.shape[0], 1)) 
        
        
        attraction1 = torch.matmul(kxy, lnpgrad_j1) 
        return  attraction1 + repulsion

    def power2(self,theta, s, lnpgrad):
        
                
                
        lnpgrad_j1 = lnpgrad[:,s]
                # calculating the kernel matrix
        kxy, repulsion = self.svgd_kernel(theta) 
        
        
        attraction1 = torch.matmul(kxy, lnpgrad_j1) 
        return  attraction1 + repulsion[:,s]

    




    def update_dim(self,score, theta,  lr):

        lnpgrad = score(theta)
                
        

            
                

        iterator_dim = range(theta.shape[1])
        s = -1
        
        for j in iterator_dim:
            if j== 1 & 2 :
                theta_blanket = theta[:,0:2]
                grad1 = self.power2(theta_blanket, s, lnpgrad)
                vec1 = 0.02*lr*grad1 / 2
                theta[:,s] = theta[:,s] + vec1.T

                


                s = s + 1
            else:
                s = s + 1
  
            
        #### STEP2 blanket_to_d
            
                grad1 = self.power(theta,  s, lnpgrad)


        
                vec1 = 0.02*lr*grad1 / 2
            
        
                theta[:,s] = theta[:,s] + vec1.T

        return theta




    # @profile
    def update(self, x0,score, n_iter = 1000,  lr=0.001):

        
        theta = x0.clone()
       
        iterator = tqdm(range(n_iter)) 
         
        
        

        for  i in iterator:
            theta= self.update_dim(score, theta,  lr)
 
        return theta



class mp_complx_SVGD():

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

    def score(self, X, **kwargs):
        X_cp = X.clone().requires_grad_(True)
        
        log_prob = self.distribution.log_prob(X_cp, **kwargs)
        
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

    def power1(self,theta, s, lnpgrad):
        
                
                
        lnpgrad_j1 = lnpgrad[:,s].view(theta.shape[0], 1)
                # calculating the kernel matrix
        kxy, repulsion = self.svgd_kernel(theta[:, s].view(theta.shape[0], 1)) 
        
        
        attraction1 = torch.matmul(kxy, lnpgrad_j1) 
        return  attraction1 + repulsion

    def update_blanket(self, theta, lnpgrad, k, lr ):
       
                
        blanket , support = self.divide(theta, k)
                       
                
        theta_ab = torch.cat([theta[:,blanket],theta[:,support]], 1)
                
        lnpgrad_blanket = lnpgrad[:,blanket]
                    # calculating the kernel matrix
        kxy, repulsion = self.svgd_kernel(theta_ab) 
            
        
        attraction = torch.matmul(kxy, lnpgrad_blanket) 
                    

        theta[:,blanket] = theta[:,blanket]  + 0.01*lr*(attraction + repulsion[:, 0:blanket.size(0)]) / theta_ab.size(0)

        return theta, blanket, support


    def update_dim(self, theta,   complx, **kwargs):

        lnpgrad = self.score(theta,**kwargs)
                
        blanket = torch.arange(0, complx, device=self.d)

            
        grad1_theta = theta.clone().detach()
        

        iterator_dim = range(theta.shape[1])
        s = -1

        for j in iterator_dim:
            s = s + 1
            
            if s in blanket:
                a = torch.where(blanket == s)[0]
    
                blanket = self.del_tensor_ele(blanket, a)
            
    
                

        #### STEP2 blanket_to_d
            
                grad1 = self.power(theta, blanket, s, lnpgrad)
                grad1_theta[:,s] =  grad1 / (blanket.shape[0] + 1)

        ### STEP3 SUPPORT_to_d
            else:
                grad2 = self.power1(theta,  s, lnpgrad)
               # print(grad2.shape)
                grad1_theta[:,s] =  grad2.squeeze()
            


        return grad1_theta



    
    

    def step(self,x0,theta,lr,complx, **kwargs):
        b = self.update_dim( theta,  complx,  **kwargs)
        opt =  optim.Adam([x0], lr=lr)
        opt.zero_grad()
        x0.grad = -b
        opt.step()
        
        theta = x0.clone()
        return theta



    # @profile
    def update(self, x0, n_iter = 1000,  lr=0.001, complx = 1, **kwargs):

        
        theta = x0.clone()
       

        
        iterator = tqdm(range(n_iter)) 
         
        
        

        for  i in iterator:
           theta =  self.step(x0,theta,lr, complx, **kwargs)
        return theta