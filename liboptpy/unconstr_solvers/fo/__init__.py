from ._gd import GradientDescent
from ._cg import ConjugateGradientFR
from ._cg import ConjugateGradientQuad
from ._bb import BarzilaiBorweinMethod
from ._acc_gd import AcceleratedGD
from ._subgrad import SubgradientMethod
from ._dual_average import DualAveraging

__all__ = ["BarzilaiBorweinMethod",
           "AcceleratedGD",
           "GradientDescent",
           "ConjugateGradientFR",
           "ConjugateGradientQuad",
           "SubgradientMethod",
           "DualAveraging"]