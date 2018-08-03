import numpy as np

__all__ = ["ConstantStepSize", "Backtracking", "ExactLineSearch4Quad", "InvIterStepSize", "ScaledInvIterStepSize"]

class StepSize(object):
    '''
    Base class for all classes for defining step size
    '''
    def __init__(self):
        pass
    def get_stepsize(self, *args, **kwargs):
        raise NotImplementedError("Method to get current step size has to be implemented!")
        
    def assign_function(self, f, grad, *args):
        pass
    
class ConstantStepSize(StepSize):
    '''
    Class represents interface for constant step size 
    '''
    def __init__(self, stepsize):
        self.stepsize = stepsize
    
    def get_stepsize(self, h, x, num_iter, *args):
        return self.stepsize
    
class ScaledConstantStepSize(StepSize):
    def __init__(self, stepsize):
        self.stepsize = stepsize
    
    def get_stepsize(self, h, x, num_iter, *args):
        return self.stepsize / np.linalg.norm(h)
    
class InvIterStepSize(StepSize):
    def __init__(self):
        pass
    
    def get_stepsize(self, h, x, num_iter, *args):
        return 1. / num_iter
    
class ScaledInvIterStepSize(StepSize):
    def __init__(self):
        pass
    
    def get_stepsize(self, h, x, num_iter, *args):
        s = 1. / num_iter
        return s / np.linalg.norm(h)
    
class InvSqrootIterStepSize(StepSize):
    def __init__(self):
        pass
    
    def get_stepsize(self, h, x, num_iter, *args):
        return 1. / np.sqrt(num_iter)
        
class Backtracking(StepSize):
    '''
    Class represents different rules for backtracking search of step size
    '''
    def __init__(self, rule_type, **kwargs):
        self.rule = rule_type
        self.par = kwargs
        if self.rule == "Lipschitz" and "eps" not in self.par:
            self.par["eps"] = 0.
        if "disp" not in self.par:
            self.par["disp"] = False
        if self.rule == "Lipschitz":
            self._alpha = None
    
    def assign_function(self, f, grad, update_x_next):
        self._f = f
        self._grad = grad
        self._update_x_next = update_x_next
    
    def get_stepsize(self, h, x, num_iter, *args):
        alpha = self.par["init_alpha"]
        if self.rule == "Armijo":
            rho = self.par["rho"]
            beta = self.par["beta"]
            assert beta < 0.5, "Armijo rule is applicable for beta less than 0.5"
            assert rho < 1, "Decay factor has to be less than 1"
            current_grad = self._grad(x)
            current_f = self._f(x)
            x_next = self._update_x_next(x, alpha, h)
            while True:
                if np.isnan(self._f(x_next)):
                    alpha *= rho
                else:
                    if self._f(x_next) >= current_f + beta * current_grad.dot(x_next - x):
                        alpha *= rho
                    else:
                        break
                if alpha < 1e-16:
                    raise ValueError("Step size is too small!")
                x_next = self._update_x_next(x, alpha, h)
            return alpha
        elif self.rule == "Wolfe":
            # https://sites.math.washington.edu/~burke/crs/408/notes/nlp/line.pdf aA
            rho = self.par["rho"]
            lb = 0
            ub = np.inf
            assert rho < 1, "Decay factor has to be less than 1"
            beta1 = self.par["beta1"]
            beta2 = self.par["beta2"]
            assert 0 < beta1 < beta2 < 1, "Wolfe rule is applicable for betas such that 0 < beta1 < beta2 < 1"
            current_grad = self._grad(x)
            current_f = self._f(x)
            while True: 
                if np.isnan(self._f(x + alpha * h)):
                    alpha *= rho
                else:
                    if self._f(x + alpha * h) > current_f + beta1 * alpha * current_grad.dot(h):
                        ub = alpha
                        alpha = 0.5 * (lb + ub)
                    elif h.dot(self._grad(x + alpha * h)) < beta2 * h.dot(current_grad):
                        lb = alpha
                        if np.isinf(ub):
                            alpha = 2 * lb
                        else:
                            alpha = 0.5 * (lb + ub)
                    else:
                        break
                if alpha < 1e-16:
                    raise ValueError("Step size is too small!")
            return alpha
        elif self.rule == "Goldstein":
            pass
        elif self.rule == "Wolfe strong":
            rho = self.par["rho"]
            assert rho < 1, "Decay factor has to be less than 1"
            beta1 = self.par["beta1"]
            beta2 = self.par["beta2"]
            lb = 0
            ub = np.inf
            assert 0 < beta1 < beta2 < 1, "Wolfe rule is applicable for betas such that 0 < beta1 < beta2 < 1"
            current_grad = self._grad(x)
            current_f = self._f(x)
            while True: 
                if np.isnan(self._f(x + alpha * h)):
                    alpha *= rho
                else:
                    if self._f(x + alpha * h) > current_f + beta1 * alpha * current_grad.dot(h):
                        ub = alpha
                        alpha = 0.5 * (lb + ub)
                    elif np.abs(h.dot(self._grad(x + alpha * h))) > beta2 * np.abs(h.dot(current_grad)):
                        lb = alpha
                        if np.isinf(ub):
                            alpha = 2 * lb
                        else:
                            alpha = 0.5 * (lb + ub)
                    else:
                        break
                if alpha < 1e-16:
                    raise ValueError("Step size is too small!")
            return alpha
        elif self.rule == "Lipschitz":
            rho = self.par["rho"]
            assert rho < 1, "Decay factor has to be less than 1"
            current_grad = self._grad(x)
            current_f = self._f(x)
            eps = self.par["eps"]
            if self._alpha is None:
                self._alpha = alpha
            else:
                self._alpha /= rho
            x_next = self._update_x_next(x, self._alpha, h)
            while True: 
                if self.par["disp"]:
                    print("Current test alpha = {}".format(self._alpha))
                if np.isnan(self._f(x_next)):
                    self._alpha *= rho
                else:
                    if self._f(x_next) > current_f + current_grad.dot(x_next - x) + np.linalg.norm(x_next - x)**2 / (2 * self._alpha) + eps:
                        self._alpha *= rho
                    else:
                        if self.par["disp"]:
                            print("Found alpha = {}".format(self._alpha))
                        break
                if self._alpha < 1e-16:
                    raise ValueError("Step size is too small!")
                x_next = self._update_x_next(x, self._alpha, h)
            return self._alpha
        else:
            raise NotImplementedError("Available rules for backtracking are 'Armijo', 'Goldstein', 'Wolfe', 'Wolfe strong' and 'Lipschitz'")

class ExactLineSearch4Quad(StepSize):
    def __init__(self, A, b=None):
        self._A = A
        if b is None:
            self._b = np.zeros(A.shape[0])
        else:
            self._b = b
    
    def get_stepsize(self, h, x, num_iter):
        return h.dot(self._b - self._A.dot(x)) / h.dot(self._A.dot(h))