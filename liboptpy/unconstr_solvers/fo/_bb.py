from ... import base_optimizer as _base
import numpy as _np

class BarzilaiBorweinMethod(_base.LineSearchOptimizer):
    def __init__(self, f, grad, **kwargs):
        super().__init__(f, grad, None, memory_size=2, **kwargs)
    
    def get_direction(self, x):
        return -self._grad(x)
    
    def get_stepsize(self):
        if len(self.convergence) == 1:
            return self._par["init_alpha"]
        else:
            g = self._grad_mem[-2] - self._grad_mem[-1]
            s = self.convergence[-1] - self.convergence[-2]
            if self._par["type"] == 1:
                alpha = g.dot(s) / g.dot(g)
            elif self._par["type"] == 2:
                alpha = s.dot(s) / g.dot(s)
            return alpha