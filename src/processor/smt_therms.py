"""
A module that processes the reviews of smart thermostats.
"""

from src.processor._amazon import _Reviews


class SmartThermostats(_Reviews):
    """
    Process the reviews of *smart thermostats*.

    This class inherits from the :class:`~src.processor._amazon._Reviews` class.
    """

    #: Name of the product.
    PRODUCT_NAME: str = 'Smart thermostats'
    #: Category of the product.
    PRODUCT_CATEGORY: str = 'Thermostats'
    #: Type of the product.
    PRODUCT_TYPE: str = 'Smart'

    #: Default column name of original review text.
    ORIGINAL_REVIEW_COLUMN_NAME: str = 'ReviewText'

    #: Schema name.
    SCHEMA_NAME: str = 'amazon_reviews'
    #: Table name.
    TABLE_NAME: str = 'thermostats_smart'
    #: Full table in PostgreSQL query statement.
    TABLE_IN_QUERY: str = f'"{SCHEMA_NAME}"."{TABLE_NAME}"'
    #: PostgreSQL query statement to read the whole table.
    SQL_QUERY: str = f'SELECT * FROM {TABLE_IN_QUERY}'

    def __init__(self, load_preprocd_data=True, **kwargs):
        # noinspection PyShadowingNames
        """
        :param load_preprocd_data: Whether to load the preprocessed data; defaults to ``False``.
        :type load_preprocd_data: bool
        :param kwargs: [Optional] parameters for initiating the class
            :class:`~src.processor._Base`

        **Examples**::

            >>> from src.processor import SmartThermostats
            >>> smt = SmartThermostats()
            >>> smt.PRODUCT_NAME
            'Smart thermostats'
            >>> smt.preprocd_data.shape
            (46317, 19)
            >>> smt = SmartThermostats(verified_reviews_only=True)
            >>> smt.preprocd_data.shape
            (38810, 19)
        """

        super().__init__(load_preprocd_data=load_preprocd_data, **kwargs)

        # smt.load()


if __name__ == '__main__':
    from src.processor import SmartThermostats
    from pyhelpers.ops import confirmed

    if confirmed("Proceed to update preprocessed data sets?"):
        load_args = {'update': True, 'verbose': True}
        for use_db in [False, True]:
            smt = SmartThermostats(load_preprocd_data=False, use_db=use_db)
            smt.load_raw_data(**load_args)
            smt.load_prep_data(**load_args)
            smt.load_preprocd_data(word_count_threshold=5, **load_args)
