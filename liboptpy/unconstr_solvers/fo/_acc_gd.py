from .. import base_optimizer as _base
import numpy as _np

class AcceleratedGD(_base.LineSearchOptimizer):
    def __init__(self, f, grad, step_size, **kwargs):
        super().__init__(f, grad, step_size, **kwargs)
        
    def get_direction(self, x):
        return -self._grad(x)
    
    def solve(self, x0, max_iter=100, tol=1e-6, disp=False):
        self.convergence = []
        x_prev = x0.copy()
        y = x0.copy()
        k = 1.
        self.convergence.append(x0)
        iteration = 0
        while True:
            h = self.get_direction(y)
            alpha = self.get_stepsize(h)
            x = y + alpha * h
            y = x + (k - 1) / (k + 2) * (x - x_prev)
            x_prev = x
            k += 1
            self.convergence.append(x)
            iteration += 1
            if disp > 1:
                print("Iteration {}/{}".format(iteration, max_iter))
                print("Current function val =", self._f(x))
                print("Current gradient norm = ", _np.linalg.norm(self._grad(x)))
            if self.check_convergence(tol) or iteration >= max_iter:
                break
        if disp:
            print("Convergence in {} iterations".format(iteration))
            print("Norm of gradient = {}".format(_np.linalg.norm(self._grad(x))))
            print("Function value = {}".format(self._f(x)))
        return x