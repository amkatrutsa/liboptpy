import numpy as _np
from ... import base_optimizer as _base

class NewtonMethod(_base.LineSearchOptimizer):
    def __init__(self, f, grad, hess, step_size, linsolver=None, **kwargs):
        super().__init__(f, grad, step_size, **kwargs)
        self._hess = hess
        self._linsolver = linsolver
    
    def get_direction(self, x):
        self._current_grad = self._grad(x)
        hess = self._hess(x)
        if self._linsolver:
            h = self._linsolver(hess, -self._current_grad)
        else:
            h = _np.linalg.solve(hess, -self._current_grad)
        return h
    
    def get_stepsize(self):
        return self._step_size.get_stepsize(self._h, self.convergence[-1], len(self.convergence))
    
