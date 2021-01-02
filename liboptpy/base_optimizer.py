import numpy as np
from collections import deque

class LineSearchOptimizer(object):
    def __init__(self, f, grad, step_size, memory_size=1, **kwargs):
        self.convergence = []
        self._f = f
        self._grad = grad
        if step_size is not None:
            step_size.assign_function(f, grad, self._f_update_x_next)
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
        self._current_grad = None
        while True:
            self._h = self.get_direction(self._x_current)
            if self._current_grad is None:
                raise ValueError("Variable self._current_grad has to be initialized in method get_direction()!")
            self._grad_mem.append(self._current_grad)
            if self.check_convergence(tol):
                if disp > 0:
                    print("Required tolerance achieved!")
                break
            if disp > 1:
                print("Iteration {}/{}".format(iteration, max_iter))
                print("Current function val =", self._f(self._x_current))
                self._print_info()
            self._alpha = self.get_stepsize()
            self._update_x_next()
            self._update_x_current()
            self._append_conv()
            iteration += 1
            if iteration >= max_iter:
                if disp > 0:
                    print("Maximum iteration exceeds!")
                break
        if disp:
            print("Convergence in {} iterations".format(iteration))
            print("Function value = {}".format(self._f(self._x_current)))
            self._print_info()
        return self._get_result_x()
    
    def get_direction(self, x):
        raise NotImplementedError("You have to provide method for finding direction!")
        
    def _update_x_current(self):
        self._x_current = self._x_next
        
    def _update_x_next(self):
        self._x_next = self._f_update_x_next(self._x_current, self._alpha, self._h)
        
    def _f_update_x_next(self, x, alpha, h):
        return x + alpha * h
        
    def check_convergence(self, tol):
        return np.linalg.norm(self._current_grad) < tol
        
    def get_stepsize(self):
        raise NotImplementedError("You have to provide method for finding step size!")
    
    def _print_info(self):
        print("Norm of gradient = {}".format(np.linalg.norm(self._current_grad)))
    
    def _append_conv(self):
        self.convergence.append(self._x_next)
        
    def _get_result_x(self):
        return self._x_current
    
class TrustRegionOptimizer(object):
    def __init__(self):
        raise NotImplementedError("Trust region methods are not implemented yet")