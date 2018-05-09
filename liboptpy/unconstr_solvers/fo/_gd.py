from .. import base_optimizer as _base

class GradientDescent(_base.DescentMethod):
    def __init__(self, f, grad, step_size, **kwargs):
        super().__init__(f, grad, step_size, **kwargs)
    
    def get_descent_direction(self, x):
        return -self._grad(x)