import numpy as np
from collections import deque

class LineSearchOptimizer(object):
    def __init__(self, f, grad, step_size, memory_size=1, **kwargs):
        self.convergence = []
        self._f = f
        self._grad = grad
        if step_size is not None:
            step_size.assign_function(f, grad)
        self._step_size = step_size
        self._par = kwargs
        self._grad_mem = deque(maxlen=memory_size)
        
    def get_convergence(self):
        return self.convergence
    
    def solve(self, x0, max_iter=100, tol=1e-6, disp=False):
        self.convergence = []
        self._x_current = x0.copy()
        self.convergence.append(self._x_current)
        iteration = 0
        while True:
            h = self.get_direction(self._x_current)
            self._grad_mem.append(h)
            alpha = self.get_stepsize()
            self._update_x_next(h, alpha)
            self._update_x_current()
            self.convergence.append(self._x_next)
            iteration += 1
            if disp > 1:
                print("Iteration {}/{}".format(iteration, max_iter))
                print("Current function val =", self._f(self._x_current))
                print("Current gradient norm = ", np.linalg.norm(self._grad(self._x_current)))
            if self.check_convergence(tol) or iteration >= max_iter:
                break
        if disp:
            print("Convergence in {} iterations".format(iteration))
            print("Norm of gradient = {}".format(np.linalg.norm(self._grad(self._x_current))))
            print("Function value = {}".format(self._f(self._x_current)))
        return self._x_current
    
    def get_direction(self, x):
        raise NotImplementedError("You have to provide method for finding direction!")
        
    def _update_x_current(self):
        self._x_current = self._x_next
        
    def _update_x_next(self, h, alpha):
        self._x_next = self._x_current + alpha * h
        
    def check_convergence(self, tol):
        return np.linalg.norm(self._grad(self.convergence[-1])) < tol
        
    def get_stepsize(self):
        raise NotImplementedError("You have to provide method for finding step size!")
    
class TrustRegionOptimizer(object):
    def __init__(self):
        raise NotImplementedError("Trust region methods are not implemented yet")