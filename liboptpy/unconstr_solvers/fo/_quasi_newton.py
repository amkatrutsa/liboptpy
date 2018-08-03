import numpy as _np
from ... import base_optimizer as _base
from ... import step_size as _ss
from collections import deque


class BFGS(_base.LineSearchOptimizer):
    
    def __init__(self, f, grad, step_size=None, 
                 H=None, **kwargs):
        if step_size is None:
            step_size = _ss.Backtracking("Wolfe", rho=0.5, beta1=1e-3, beta2=0.9, init_alpha=1.)
        super().__init__(f, grad, step_size, memory_size=1, **kwargs)
        self._H0 = H
        self._H = H
    
    def get_direction(self, x):
        if self._H is None:
            self._current_grad = self._grad(x)
            return -self._current_grad
        else:
            return -self._H.dot(self._current_grad)
    
    def get_stepsize(self):
        return self._step_size.get_stepsize(self._h, self.convergence[-1], len(self.convergence))
    
    def _update_x_current(self):
        self._current_grad = self._grad(self._x_next)
        s = self._x_next - self._x_current
        y = self._current_grad - self._grad_mem[-1]
        rho = 1. / y.dot(s)
        if self._H is None:
            self._H = _np.eye(self._x_current.shape[0]) / y.dot(y) / rho
        Hy = self._H.dot(y)
        Hys = _np.outer(Hy, s)
        ss = _np.outer(s, s)
        self._H = rho * ss + self._H - rho * Hys - rho * Hys.T + \
                  rho**2 * y.dot(Hy) * ss
        self._x_current = self._x_next
        
    def _get_result_x(self):
        self._H = self._H0
        return self._x_current
    
    
class LBFGS(_base.LineSearchOptimizer):
    
    def __init__(self, f, grad, step_size=None, 
                 H=None, hist_size=10, **kwargs):
        if step_size is None:
            step_size = _ss.Backtracking("Wolfe", rho=0.5, beta1=1e-3, beta2=0.9, init_alpha=1.)
        super().__init__(f, grad, step_size, memory_size=1, **kwargs)
        self._H0 = H
        self._H = H
        self._s_hist = deque(maxlen=hist_size)
        self._y_hist = deque(maxlen=hist_size)
    
    def get_direction(self, x):
        if self._H is None:
            self._current_grad = self._grad(x)
            return -self._current_grad
        else:
            q = self._current_grad
            alpha = _np.zeros(len(self._s_hist))
            rho = _np.zeros(len(self._s_hist))
            for i in range(len(self._s_hist) - 1, -1, -1):
                rho[i] = 1. / self._s_hist[i].dot(self._y_hist[i])
                alpha[i] = self._s_hist[i].dot(q) * rho[i]
                q = q - alpha[i] * self._y_hist[i]
            r = q * self._H
            for i in range(len(self._s_hist)):
                beta = rho[i] * self._y_hist[i].dot(r)
                r = r + self._s_hist[i] * (alpha[i] - beta)
            return -r
    
    def get_stepsize(self):
        return self._step_size.get_stepsize(self._h, self.convergence[-1], len(self.convergence))
    
    def _update_x_current(self):
        self._current_grad = self._grad(self._x_next)
        s = self._x_next - self._x_current
        y = self._current_grad - self._grad_mem[-1]
        self._s_hist.append(s)
        self._y_hist.append(y)
        self._H = y.dot(s) / y.dot(y)
        self._x_current = self._x_next
        
    def _get_result_x(self):
        self._H = self._H0
        return self._x_current
    
    
class DFP(_base.LineSearchOptimizer):
    
    def __init__(self, f, grad, step_size=None, 
                 H=None, **kwargs):
        if step_size is None:
            step_size = _ss.Backtracking("Wolfe", rho=0.5, beta1=1e-3, beta2=0.9, init_alpha=1.)
        super().__init__(f, grad, step_size, memory_size=2, **kwargs)
        self._H0 = H
        self._H = H
    
    def get_direction(self, x):
        if self._H is None:
            self._current_grad = self._grad(x)
            return -self._current_grad
        else:
            return -self._H.dot(self._current_grad)
    
    def get_stepsize(self):
        return self._step_size.get_stepsize(self._h, self.convergence[-1], len(self.convergence))
    
    def _update_x_current(self):
        self._current_grad = self._grad(self._x_next)
        s = self._x_next - self._x_current
        y = self._current_grad - self._grad_mem[-1]
        rho = 1. / y.dot(s)
        if self._H is None:
            self._H = _np.eye(self._x_current.shape[0]) / y.dot(y) / rho
        Hy = self._H.dot(y)
        self._H = self._H - ((_np.outer(Hy, Hy)) / (y.dot(Hy))) + (_np.outer(s, s) * rho)
        self._x_current = self._x_next
        
    def _get_result_x(self):
        self._H = self._H0
        return self._x_current
    
class BarzilaiBorweinMethod(_base.LineSearchOptimizer):
    def __init__(self, f, grad, **kwargs):
        super().__init__(f, grad, None, memory_size=2, **kwargs)
    
    def get_direction(self, x):
        self._current_grad = self._grad(x)
        return -self._current_grad
    
    def get_stepsize(self):
        if len(self.convergence) == 1:
            return self._par["init_alpha"]
        else:
            g = self._grad_mem[-1] - self._grad_mem[-2]
            s = self.convergence[-1] - self.convergence[-2]
            if self._par["type"] == 1:
                alpha = g.dot(s) / g.dot(g)
            elif self._par["type"] == 2:
                alpha = s.dot(s) / g.dot(s)
            return alpha