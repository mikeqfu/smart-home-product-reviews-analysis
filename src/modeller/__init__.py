"""
The module is used for applying algorithms on the data generated from :mod:`~src.processor`.
"""

import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

from ._base import _Base
from .lda import LatentDirichletAllocation
from .lrm import LogisticRegressionModel
