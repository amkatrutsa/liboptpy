from ... import base_optimizer as _base
import numpy as _np

class ConjugateGradientFR(_base.LineSearchOptimizer):
    def __init__(self, f, grad, step_size, restart=None, **kwargs):
        super().__init__(f, grad, step_size, **kwargs)
        if restart is not None:
            restart.assign_function(f, grad)
        self._restart = restart
        
    def get_direction(self, x):
        if (len(self.convergence) == 1) or (self._restart is not None and 
                                            self._restart(len(self.convergence), x)):
            self._current_grad = self._grad(x)
            h = -self._current_grad
        else:
            self._current_grad = self._grad(self.convergence[-1])
            beta = self._current_grad.dot(self._current_grad) / self._h.dot(self._h)
            h = -self._current_grad + beta * self._h
        return h
    
    def get_stepsize(self):
        return self._step_size.get_stepsize(self._h, self.convergence[-1], len(self.convergence)) 
    
class ConjugateGradientQuad(_base.LineSearchOptimizer):
    def __init__(self, A, b=None):
        if b is None:
            b = _np.zeros(A.shape[0])
        f = lambda x: 0.5 * x.dot(A.dot(x)) - b.dot(x)
        grad = lambda x: A.dot(x) - b
        super().__init__(f, grad, None)
        self._A = A
        self._b = b
        
    def get_direction(self, x):
        if (len(self.convergence) == 1):
            self._current_grad = self._grad(x)
            h = -self._current_grad
            self._r = -h
        else:
            r_next = self._r + self._alpha * self._A.dot(self._h)
            beta = r_next.dot(r_next) / self._r.dot(self._r)
            h = -r_next + beta * self._h
            self._r = r_next
        return h
    
    def get_stepsize(self):
        # h = self._grad_mem[-1]
        self._alpha = self._r.dot(self._r) / self._h.dot(self._A.dot(self._h))
        return self._alpha
    
    def check_convergence(self, tol):
        return _np.linalg.norm(self._grad(self.convergence[-1])) < tol
    