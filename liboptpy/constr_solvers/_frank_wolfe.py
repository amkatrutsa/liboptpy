import numpy as np
from ..base_optimizer import LineSearchOptimizer

class FrankWolfe(LineSearchOptimizer):
    
    '''
    Class represents conditional gradient descent method aka Frank Wolfe algorithm
    '''
    
    def __init__(self, f, grad, linsolver, step_size):
        super().__init__(f, grad, step_size)
        self._linsolver = linsolver
        self._h = None
        
    def get_direction(self, x):
        s = self._linsolver(self._grad(x))
        self._current_grad = self._grad(x)
        self._h = s - x
        return self._h
    
    def check_convergence(self, tol):
        if len(self.convergence) == 1:
            return False
        if self._f(self.convergence[-2]) - self._f(self.convergence[-1]) < tol:
            return True
        else:
            return False
        
    def get_stepsize(self):
        return self._step_size.get_stepsize(self._h, self.convergence[-1], len(self.convergence))
    
    def _print_info(self):
        print("Difference in function values = {}".format(self._f(self.convergence[-2]) - self._f(self.convergence[-1])))
        print("Difference in argument = {}".format(np.linalg.norm(self.convergence[-1] - self.convergence[-2])))
