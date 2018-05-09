import numpy as _np
from .. import base_optimizer as _base

class NewtonMethod(_base.DescentMethod):
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
            h = _np.linalg.solve(hess, -grad)
        return h
    
