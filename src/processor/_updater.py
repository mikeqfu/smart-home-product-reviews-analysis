"""
A module for updating/(pre-)processing the data of review texts.
"""

from pyhelpers.ops import confirmed

from src.processor import *

if __name__ == '__main__':
    if confirmed("Proceed to update preprocessed data sets?"):
        load_args = {'update': True, 'verbose': True}

        for use_db in [False, True]:
            for cls in [RoboticVacuumCleaners, TraditionalVacuumCleaners, SmartThermostats]:
                word_count_threshold = 20
                if cls == SmartThermostats:
                    word_count_threshold = 5

                cls_instance = cls(load_preprocd_data=False, use_db=use_db)
                cls_instance.load_raw_data(**load_args)  # Update raw_data
                cls_instance.load_prep_data(**load_args)  # Update prep_data
                cls_instance.load_preprocd_data(
                    word_count_threshold=word_count_threshold, **load_args)  # Update preprocd_data
