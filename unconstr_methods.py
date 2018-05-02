import numpy as np

__all__ = ["GradientDescent", "Newton"]

class DescentMethod(object):
    def __init__(self, f, grad, step_size, **kwargs):
        self.convergence = []
        self._f = f
        self._grad = grad
        step_size.assign_function(f, grad)
        self._step_size = step_size
        self._par = kwargs
        
    def get_convergence(self):
        return self.convergence
    
    def solve(self, x0, max_iter=100, tol=1e-6, disp=False):
        self.convergence = []
        x = x0.copy()
        self.convergence.append(x)
        iteration = 0
        while True:
            h = self.get_descent_direction(x)
            alpha = self.get_stepsize(h)
            x = x + alpha * h
            self.convergence.append(x)
            iteration += 1
            if disp > 1:
                print("Iteration {}/{}".format(iteration, max_iter))
                print("Current function val =", self._f(x))
                print("Current gradient norm = ", np.linalg.norm(self._grad(x)))
            if self.check_convergence(tol) or iteration >= max_iter:
                break
        if disp:
            print("Convergence in {} iterations".format(iteration))
            print("Norm of gradient = {}".format(np.linalg.norm(self._grad(x))))
            print("Function value = {}".format(self._f(x)))
        return x
    
    def get_descent_direction(self, x):
        raise NotImplementedError("You have to provide method for descent direction!")
        
    def check_convergence(self, tol):
        return np.linalg.norm(self._grad(self.convergence[-1])) < tol
        
    def get_stepsize(self, h):
        return self._step_size.get_stepsize(h, self.convergence[-1], **self._par)
        
class GradientDescent(DescentMethod):
    def __init__(self, f, grad, step_size, **kwargs):
        super().__init__(f, grad, step_size, **kwargs)
    
    def get_descent_direction(self, x):
        return -self._grad(x)
    
    
class Newton(DescentMethod):
    def __init__(self, f, grad, hess, step_size, linsolver=None, **kwargs):
        super().__init__(f, grad, step_size, **kwargs)
        self._hess = hess
        self._linsolver = linsolver
    
    def get_descent_direction(self, x):
        grad = self._grad(x)
        hess = self._hess(x)
        if self._linsolver:
            h = self._linsolver(hess, -grad)
        else:
            h = np.linalg.solve(hess, -grad)
        return h