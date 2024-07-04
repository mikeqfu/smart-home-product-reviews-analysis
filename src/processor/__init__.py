"""
The module is used for (pre-)processing (and I/O management of) all the data resources.
"""

from ._amazon import _Reviews
from .robo_vac import RoboticVacuumCleaners
from .smt_therms import SmartThermostats
from .trad_vac import TraditionalVacuumCleaners

__all__ = [
    '_Reviews',
    'RoboticVacuumCleaners',
    'TraditionalVacuumCleaners',
    'SmartThermostats']
