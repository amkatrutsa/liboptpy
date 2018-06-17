import numpy as np
from ..base_optimizer import LineSearchOptimizer
from ..step_size import ProjectedArmijo

class ProjectedGD(LineSearchOptimizer):
    
    '''
    Class represents projected gradient method
    '''
    
    def __init__(self, f, grad, projector, step_size):
        super().__init__(f, grad, step_size)
        self._projector = projector
        
    def get_direction(self, x):
        return -self._grad(x)
    
    def _f_update_x_next(self, x, alpha, h):
        return self._projector(x + alpha * h)
    
    def check_convergence(self, tol):
        if self._f(self.convergence[-2]) - self._f(self.convergence[-1]) < tol:
            return True
        else:
            return False
        
    def get_stepsize(self):
        return self._step_size.get_stepsize(self._grad_mem[-1], self.convergence[-1], len(self.convergence))
    
    def _print_info(self):
        print("Difference in function values = {}".format(self._f(self.convergence[-2]) - self._f(self.convergence[-1])))
        print("Difference in argument = {}".format(np.linalg.norm(self.convergence[-1] - self.convergence[-2])))