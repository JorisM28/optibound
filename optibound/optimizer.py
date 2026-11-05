import numpy as np





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


        def minimize(x_init):

            x_init = torch.tensor([8.0, 8.0])

            self.history["status"] = "active"
            self.history["x"] = x_init
            self.history["loss"] = self.func(x_init)

            grad_norm = torch.linalg.vector_norm(self.func.gradient(x_init).item()
            self.history["grad_norm"] = grad
            
            for _ in range(max_iter):






                


                
