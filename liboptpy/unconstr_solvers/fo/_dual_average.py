from ... import base_optimizer as _base
import numpy as _np

class DualAveraging(_base.LineSearchOptimizer):
    def __init__(self, f, subgrad, primal_step_size, dual_step_size):
        super().__init__(f, subgrad, primal_step_size)
        self._dual_step_size = dual_step_size
        self._sum_lam = 0
        
    def get_direction(self, x):
        self._current_grad = self._grad(x)
        if len(self.convergence) == 1:
            self._s = _np.zeros(x.shape[0])
        self._lam = self._dual_step_size.get_stepsize(x, self._current_grad, len(self.convergence))
        self._s = (self._sum_lam * self._s + self._lam * self._current_grad) / (self._sum_lam + self._lam)
        self._sum_lam += self._lam
        return -self._s
    
    def get_stepsize(self):
        return self._step_size.get_stepsize(self._h, self._x_current, len(self.convergence))
        
    def _f_update_x_next(self, x, alpha, h):
        return self.convergence[0] + alpha * h
    
    def _append_conv(self):
        self.convergence.append(self._x_current)
        
    def _update_x_current(self):
        self._x_current = self._x_next