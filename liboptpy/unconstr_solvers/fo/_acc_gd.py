from ... import base_optimizer as _base
import numpy as _np
from ... import step_size as ss

class AcceleratedGD(_base.LineSearchOptimizer):
    def __init__(self, f, grad, step_size, momentum_size=None, **kwargs):
        super().__init__(f, grad, step_size, **kwargs)
        if momentum_size is not None:
            momentum_size.assign(f, grad)
        self._momentum_size = momentum_size
        self._lam0 = 0
        self._lam1 = 1
        
    def get_direction(self, x):
        self._current_grad = self._grad(x)
        return -self._current_grad
    
    def _update_x_current(self):
        if self._momentum_size is None:
            beta = (self._lam0 - 1) / self._lam1
            t = self._lam0
            self._lam0 = self._lam1
            self._lam1 = (1 + _np.sqrt(1 + 4 * t**2)) / 2.
        self._x_current = self._x_next + beta * (self._x_next - self.convergence[-1])
        
    def get_stepsize(self):
        return self._step_size.get_stepsize(self._h, self.convergence[-1], len(self.convergence))