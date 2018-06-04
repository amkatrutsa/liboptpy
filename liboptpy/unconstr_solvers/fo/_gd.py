from ... import base_optimizer as _base

class GradientDescent(_base.LineSearchOptimizer):
    def __init__(self, f, grad, step_size, **kwargs):
        super().__init__(f, grad, step_size, **kwargs)
    
    def get_direction(self, x):
        return -self._grad(x)
    
    def get_stepsize(self):
        return self._step_size.get_stepsize(self._grad_mem[-1], self.convergence[-1], len(self.convergence))