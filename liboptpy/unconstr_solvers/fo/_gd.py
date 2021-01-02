from ... import base_optimizer as _base

class GradientDescent(_base.LineSearchOptimizer):
    def __init__(self, f, grad, step_size, **kwargs):
        super().__init__(f, grad, step_size, **kwargs)
    
    def get_direction(self, x):
        self._current_grad = self._grad(x)
        return -self._current_grad
    
    def get_stepsize(self):
        return self._step_size.get_stepsize(self._h, self.convergence[-1], len(self.convergence))