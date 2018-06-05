import numpy as np
from ... import base_optimizer as base
from ..fo import _cg as cg

class InexactNewtonMethod(base.LineSearchOptimizer):
    def __init__(self, f, grad, hess_matvec, step_size, **kwargs):
        super().__init__(f, grad, step_size, **kwargs)
        self._hess_matvec = hess_matvec
    
    def get_direction(self, x):
        grad = self._grad(x)
        hess = self._hess_matvec(x)
        lin_cg = cg.ConjugateGradientQuad(hess, -grad)
        eta = np.minimum(0.5, np.sqrt(np.linalg.norm(grad)))
        h = np.zeros(grad.shape[0])
        while True:
            h = lin_cg.solve(x0=h, tol=eta)
            if h.dot(grad) < 0:
                break
            else:
                eta = eta / 10.
        
        return h
    
    def get_stepsize(self):
        return self._step_size.get_stepsize(self._grad_mem[-1], self.convergence[-1], len(self.convergence))