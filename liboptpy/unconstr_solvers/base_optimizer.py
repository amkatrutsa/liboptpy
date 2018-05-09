import numpy as np

class LineSearchOptimizer(object):
    def __init__(self, f, grad, step_size, **kwargs):
        self.convergence = []
        self._f = f
        self._grad = grad
        if step_size is not None:
            step_size.assign_function(f, grad)
        self._step_size = step_size
        self._par = kwargs
        
    def get_convergence(self):
        return self.convergence
    
    def solve(self, x0, max_iter=100, tol=1e-6, disp=False):
        self.convergence = []
        x = x0.copy()
        self.convergence.append(x)
        iteration = 0
        while True:
            h = self.get_direction(x)
            alpha = self.get_stepsize(h)
            x = x + alpha * h
            self.convergence.append(x)
            iteration += 1
            if disp > 1:
                print("Iteration {}/{}".format(iteration, max_iter))
                print("Current function val =", self._f(x))
                print("Current gradient norm = ", np.linalg.norm(self._grad(x)))
            if self.check_convergence(tol) or iteration >= max_iter:
                break
        if disp:
            print("Convergence in {} iterations".format(iteration))
            print("Norm of gradient = {}".format(np.linalg.norm(self._grad(x))))
            print("Function value = {}".format(self._f(x)))
        return x
    
    def get_direction(self, x):
        raise NotImplementedError("You have to provide method for finding direction!")
        
    def check_convergence(self, tol):
        return np.linalg.norm(self._grad(self.convergence[-1])) < tol
        
    def get_stepsize(self, h):
        return self._step_size.get_stepsize(h, self.convergence[-1], len(self.convergence), **self._par)
    
class TrustRegionOptimizer(object):
    def __init__(self):
        raise NotImplementedError("Trust region methods are not implemented yet")