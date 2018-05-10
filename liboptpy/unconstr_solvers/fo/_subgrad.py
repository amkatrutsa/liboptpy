from .. import base_optimizer as _base

class SubgradientMethod(_base.LineSearchOptimizer):
    def __init__(self, f, subgrad, step_size):
        super().__init__(f, subgrad, step_size)
    
    def get_direction(self, x):
        return -self._grad(x)
    
    def check_convergence(self, tol):
        return False
    
    def get_stepsize(self):
        return self._step_size.get_stepsize(self._grad_mem[-1], self._x_current, len(self.convergence))