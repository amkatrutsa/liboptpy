from .. import base_optimizer as _base
import numpy as _np

class BarzilaiBorweinMethod(_base.LineSearchOptimizer):
    def __init__(self, f, grad, **kwargs):
        super().__init__(f, grad, None, **kwargs)
    
    def get_direction(self, x):
        return -self._grad(x)
    
    def solve(self, x0, max_iter=100, tol=1e-6, disp=False):
        self.convergence = []
        x_prev = x0.copy()
        self.convergence.append(x_prev)
        iteration = 0
        while True:
            h = self.get_direction(x_prev)
            if iteration == 0:
                alpha = self._par["init_alpha"]
            else:
                alpha = self.get_stepsize(s, h_prev - h, self._par["type"])
            x_next = x_prev + alpha * h
            h_prev = h.copy()
            s = x_next - x_prev
            x_prev = x_next
            self.convergence.append(x_next)
            iteration += 1
            if disp > 1:
                print("Iteration {}/{}".format(iteration, max_iter))
                print("Current function val =", self._f(x_next))
                print("Current gradient norm = ", _np.linalg.norm(self._grad(x_next)))
            if self.check_convergence(tol) or iteration >= max_iter:
                break
        if disp:
            print("Convergence in {} iterations".format(iteration))
            print("Norm of gradient = {}".format(_np.linalg.norm(self._grad(x_next))))
            print("Function value = {}".format(self._f(x_next)))
        return x_next
    
    def get_stepsize(self, s, g, alpha_type):
        if alpha_type == 1:
            alpha = g.dot(s) / g.dot(g)
        elif alpha_type == 2:
            alpha = s.dot(s) / g.dot(s)
        return alpha