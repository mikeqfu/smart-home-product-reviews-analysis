"""
A module that processes the reviews of traditional vacuum cleaners.
"""

from src.processor._amazon import _Reviews


class TraditionalVacuumCleaners(_Reviews):
    """
    Process the reviews of *traditional vacuum cleaners*.

    This class inherits from the :class:`~src.processor._amazon._Reviews` class.
    """

    #: Name of the product.
    PRODUCT_NAME: str = 'Traditional vacuum cleaners'
    #: Category of the product.
    PRODUCT_CATEGORY: str = 'Vacuum cleaners'
    #: Type of the product.
    PRODUCT_TYPE: str = 'Traditional'

    #: Default column name of original review text.
    ORIGINAL_REVIEW_COLUMN_NAME: str = 'ReviewText'

    #: Schema name.
    SCHEMA_NAME: str = 'amazon_reviews'
    #: Table name.
    TABLE_NAME: str = 'vacuum_cleaners_traditional'
    #: Full table in PostgreSQL query statement.
    TABLE_IN_QUERY: str = f'"{SCHEMA_NAME}"."{TABLE_NAME}"'
    #: PostgreSQL query statement to read the whole table.
    SQL_QUERY: str = f'SELECT * FROM {TABLE_IN_QUERY}'

    def __init__(self, load_preprocd_data=True, load_prep_data=False, load_raw_data=False,
                 **kwargs):
        """
        :param load_preprocd_data: Whether to load the preprocessed data; defaults to ``False``.
        :type load_preprocd_data: bool
        :param load_prep_data: Whether to load the preparatory data; defaults to ``False``.
        :type load_prep_data: bool
        :param load_raw_data: Whether to load the raw data; defaults to ``False``.
        :type load_raw_data: bool
        :param kwargs: [Optional] parameters for initiating the class
            :class:`~src.processor._Base`.

        **Examples**::

            >>> from src.processor import TraditionalVacuumCleaners
            >>> tvc = TraditionalVacuumCleaners()
            Loading "data\\amazon_reviews\\vacuum_cleaners\\traditional\\preprocd_data\\preprocd_...
            >>> tvc.PRODUCT_NAME
            'Traditional vacuum cleaners'
            >>> tvc.preprocd_data.shape
            (146656, 19)
            >>> tvc = TraditionalVacuumCleaners(verified_reviews_only=True)
            Loading "data\\amazon_reviews\\vacuum_cleaners\\traditional\\preprocd_data\\preprocd_...
            >>> tvc.preprocd_data.shape
            (131971, 19)
        """

        super().__init__(
            load_preprocd_data=load_preprocd_data, load_prep_data=load_prep_data,
            load_raw_data=load_raw_data, **kwargs)
