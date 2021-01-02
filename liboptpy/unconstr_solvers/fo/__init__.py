from ._gd import GradientDescent
from ._cg import ConjugateGradientFR
from ._cg import ConjugateGradientQuad
from ._acc_gd import AcceleratedGD
from ._subgrad import SubgradientMethod
from ._dual_average import DualAveraging
from ._quasi_newton import BFGS, LBFGS, DFP, BarzilaiBorweinMethod

__all__ = ["BarzilaiBorweinMethod",
           "AcceleratedGD",
           "GradientDescent",
           "ConjugateGradientFR",
           "ConjugateGradientQuad",
           "SubgradientMethod",
           "DualAveraging",
           "BFGS", "LBFGS", "DFP"]