"""
Initialization.
"""

import datetime
import json
import pkgutil

from .modeller import LatentDirichletAllocation, LogisticRegressionModel
from .processor import RoboticVacuumCleaners, SmartThermostats, TraditionalVacuumCleaners

metadata = json.loads(pkgutil.get_data(__name__, "data/metadata").decode())

__project__ = metadata['Project']
__authors__ = metadata['Collaborators']
__desc__ = metadata['Description']

__copyright__ = f'2022-{datetime.datetime.now().year}, ' + ' & '.join(
    ', '.join([__authors__[f'Collaborator {i}']['Name'] for i in range(1, 4)]).rsplit(', ', 1))

__version__ = metadata['Version']
__license__ = metadata['License']
__kickoff__ = metadata['Project Start']

__all__ = [
    'processor', 'modeller', 'analyser',
    'RoboticVacuumCleaners', 'TraditionalVacuumCleaners', 'SmartThermostats',
    'LogisticRegressionModel', 'LatentDirichletAllocation',
]
