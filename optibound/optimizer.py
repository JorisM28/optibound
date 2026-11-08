import numpy as np
import torch
import math




class GradientDescent:


    def __init__(self, func, lr, max_iter, tol, divergence_thresold):

        self.func = func
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.divergence_thresold = divergence_thresold


        self.history = {
            "x" : [],
            "loss": [],
            "grad_norm": [],
            "status": None
        }


    def step(self, x):

        if x.ndim == 1:
            x_batch = x.unsqueeze(0)
        else:
            x_batch = x

        grad = self.func.gradient(x_batch).detach()

        x_next = x_batch - self.lr * grad

        return x_next.squeeze(0), grad.squeeze(0)


    def minimize(self, x_init):

        if not isinstance(x_init, torch.Tensor):
            x = torch.tensor(x_init, dtype=torch.float32)
        else:
            x = x_init.clone().detach().to(dtype=torch.float32)


        loss_init = self.func(x.unsqueeze(0)).item()
        grad_init = self.func.gradient(x.unsqueeze(0)).detach()
        grad_norm_init = torch.linalg.vector_norm(grad_init).item()


        self.history["x"].append(x.clone())
        self.history["loss"].append(loss_init)
        self.history["grad_norm"].append(grad_norm_init)
        self.history["status"] = "running"


        for _ in range(self.max_iter):
            
            x_next, grad = self.step(x)
            x = x_next  
            
            loss_val = self.func(x.unsqueeze(0)).item()
            grad_norm = torch.linalg.vector_norm(grad).item()
            
            self.history["x"].append(x.clone().detach())
            self.history["loss"].append(loss_val)
            self.history["grad_norm"].append(grad_norm)
            

            if loss_val > self.divergence_thresold or math.isnan(loss_val) or math.isinf(loss_val):
                self.history["status"] = "diverged"
                break
                

            if grad_norm < self.tol:
                self.history["status"] = "converged"
                break


        if self.history["status"] == "running":
            self.history["status"] = "max_iter_reached"

        return x














                






                


                
