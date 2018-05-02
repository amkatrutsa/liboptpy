import numpy as np

__all__ = ["GradientDescent"]

class _DescentMethod(object):
    def __init__(self):
        self.convergence = []
    
    def get_convergence(self):
        return self.convergence
    
    def solve(self, x0, max_iter=100, tol=1e-6):
        raise NotImplementedError("Method solve has to be implemented!")
    
    def get_descent_direction(self, x):
        raise NotImplementedError("You have to provide method for descent direction!")
        
    def check_convergence(self, tol):
        raise NotImplementedError("You have to stopping criterion to check convergence!")
        
    def get_stepsize(self, h):
        raise NotImplementedError("You have to provide method to set stepsize!")
        
class GradientDescent(_DescentMethod):
    def __init__(self, f, grad, step_size, **kwargs):
        super().__init__()
        self.f = f
        self.grad = grad
        self.par = kwargs
        step_size.assign_function(f, grad)
        self.step_size = step_size
        
    def check_convergence(self, tol):
        return np.linalg.norm(self.grad(self.convergence[-1])) < tol
    
    def get_descent_direction(self, x):
        return -self.grad(x)
    
    def get_stepsize(self, h):
        return self.step_size(h, self.convergence[-1], **self.par)
    
    def solve(self, x0, max_iter=100, tol=1e-6, disp=False):
        self.convergence = []
        x = x0.copy()
        self.convergence.append(x)
        iteration = 0
        while True:
            h = self.get_descent_direction(x)
            alpha = self.step_size.get_stepsize(h, x)
            x = x + alpha * h
            self.convergence.append(x)
            iteration += 1
            if disp > 1:
                print("Iteration {}/{}".format(iteration, max_iter))
                print("Current function val =", self.f(x))
                print("Current gradient norm = ", np.linalg.norm(self.grad(x)))
            if self.check_convergence(tol) or iteration >= max_iter:
                break
        if disp:
            print("Convergence in {} iterations".format(iteration))
            print("Norm of gradient = {}".format(np.linalg.norm(self.grad(x))))
            print("Function value = {}".format(self.f(x)))
        return x