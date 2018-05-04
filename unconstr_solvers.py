import numpy as np

__all__ = ["GradientDescent", "Newton", "BB_method", "ConjugateGradientQuad",
           "ConjugateGradientFR"]

class DescentMethod(object):
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
            h = self.get_descent_direction(x)
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
    
    def get_descent_direction(self, x):
        raise NotImplementedError("You have to provide method for descent direction!")
        
    def check_convergence(self, tol):
        return np.linalg.norm(self._grad(self.convergence[-1])) < tol
        
    def get_stepsize(self, h):
        return self._step_size.get_stepsize(h, self.convergence[-1], **self._par)
        
class GradientDescent(DescentMethod):
    def __init__(self, f, grad, step_size, **kwargs):
        super().__init__(f, grad, step_size, **kwargs)
    
    def get_descent_direction(self, x):
        return -self._grad(x)
    
    
class Newton(DescentMethod):
    def __init__(self, f, grad, hess, step_size, linsolver=None, **kwargs):
        super().__init__(f, grad, step_size, **kwargs)
        self._hess = hess
        self._linsolver = linsolver
    
    def get_descent_direction(self, x):
        grad = self._grad(x)
        hess = self._hess(x)
        if self._linsolver:
            h = self._linsolver(hess, -grad)
        else:
            h = np.linalg.solve(hess, -grad)
        return h
    
class BarzilaiBorwein(DescentMethod):
    def __init__(self, f, grad, **kwargs):
        super().__init__(f, grad, None, **kwargs)
    
    def get_descent_direction(self, x):
        return -self._grad(x)
    
    def solve(self, x0, max_iter=100, tol=1e-6, disp=False):
        self.convergence = []
        x_prev = x0.copy()
        self.convergence.append(x_prev)
        iteration = 0
        while True:
            h = self.get_descent_direction(x_prev)
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
                print("Current gradient norm = ", np.linalg.norm(self._grad(x_next)))
            if self.check_convergence(tol) or iteration >= max_iter:
                break
        if disp:
            print("Convergence in {} iterations".format(iteration))
            print("Norm of gradient = {}".format(np.linalg.norm(self._grad(x_next))))
            print("Function value = {}".format(self._f(x_next)))
        return x_next
    
    def get_stepsize(self, s, g, alpha_type):
        if alpha_type == 1:
            alpha = g.dot(s) / g.dot(g)
        elif alpha_type == 2:
            alpha = s.dot(s) / g.dot(s)
        return alpha
    
class ConjugateGradientQuad(DescentMethod):
    def __init__(self, A, b):
        f = lambda x: 0.5 * x.dot(A.dot(x)) - b.dot(x)
        grad = lambda x: A.dot(x) - b
        super().__init__(f, grad, None)
        self._A = A
        self._b = b
        
    def get_descent_direction(self, x):
        if (len(self.convergence) == 1):
            h = -self._grad(x)
            self._h = h
            self._r = -h
        else:
            r_next = self._r + self._alpha * self._A.dot(self._h)
            beta = r_next.dot(r_next) / self._r.dot(self._r)
            h = -r_next + beta * self._h
            self._r = r_next
            self._h = h
        return h
    
    def get_stepsize(self, h):
        alpha = self._r.dot(self._r) / h.dot(self._A.dot(h))
        self._alpha = alpha
        return alpha
    
class ConjugateGradientFR(DescentMethod):
    def __init__(self, f, grad, step_size, restart=None, **kwargs):
        super().__init__(f, grad, step_size, **kwargs)
        if restart is not None:
            restart.assign_function(f, grad)
        self._restart = restart
        
    def get_descent_direction(self, x):
        if (len(self.convergence) == 1) or (self._restart is not None and 
                                            self._restart(len(self.convergence), x)):
            h = -self._grad(x)
            self._h = h
        else:
            current_grad = self._grad(self.convergence[-1])
            beta = current_grad.dot(current_grad) / self._h.dot(self._h)
            h = -current_grad + beta * self._h
            self._h = h
        return h