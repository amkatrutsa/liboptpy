class Restart(object):
    def __init__(self, limit_dim=None):
        self._dim = limit_dim
    
    def __call__(self, num_iter, x):
        if num_iter % self._dim == 0:
            return True
        else:
            return False
    
    def assign_function(self, f, grad):
        self._f = f
        self._grad = grad
