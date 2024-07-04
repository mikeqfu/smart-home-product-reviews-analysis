"""
A module containing base classes for modelling.
"""

from pyhelpers.text import find_similar_str


class _Base:
    """
    A base class for modelling trials.
    """

    #: Valid names of a product category.
    PRODUCT_CATEGORIES: str = {'Vacuum cleaners', 'Thermostats'}
    #: Valid types of a product.
    PRODUCT_TYPES: set = {'Robotic', 'Traditional', 'Smart'}
    #: Column name of the review texts.
    REVIEW_COLUMN_NAME: str = 'review_text'
    #: Valid sentiment labels.
    VALID_SENTIMENT_LABELS: set = {'positive', 'negative', 'neutral'}

    def __init__(self, product_category, product_type, sentiment_on='dual_scale',
                 review_column_name=None, random_state=0, **kwargs):
        """
        :param product_category: Product category.
        :type product_category: str
        :param product_type: product type, valid values include ``{'Robotic', 'Traditional'}``
        :type product_type: str
        :param sentiment_on: column name for the metric on which sentiment is determined,
            defaults to ``'dual_scale'``
        :type sentiment_on: str
        :param review_column_name: column name of the review texts;
            when ``review_column_name=None`` (default), it defaults to ``'review_text'``
        :type review_column_name: str | None
        :param random_state: random seed number, defaults to ``0``
        :type random_state: int | None
        :param kwargs: [optional] parameters for initiating the class
            :class:`~src.processor._Base`

        :ivar int | None random_state: A random seed number.
        :ivar str product_type: The type of product.
        :ivar RoboticVacuumCleaners | TraditionalVacuumCleaners reviews:
            An instance of the class :class:`~src.processor.RoboticVacuumCleaners` or
            :class:`~src.processor.TraditionalVacuumCleaners`.
        :ivar str sentiment_column_name: Column name of the sentiment.
        :ivar pandas.DataFrame data: Preprocessed data of the reviews.
        :ivar str review_column_name: Column name of the review texts.
        """

        self.random_state = random_state

        self.product_category = find_similar_str(
            product_category, self.PRODUCT_CATEGORIES, engine='fuzz')
        self.product_type = find_similar_str(product_type, self.PRODUCT_TYPES, engine='fuzz')

        if self.product_type is None:
            raise ValueError(f"Invalid `product_type`. Choose from {self.PRODUCT_TYPES}.")
        else:
            mod = __import__('src.processor')
            self.reviews = getattr(
                mod, f'{self.product_type}{self.product_category.title().replace(" ", "")}')(
                **kwargs)

        self.sentiment_column_name = find_similar_str(
            sentiment_on, self.reviews.sentiment_column_names, engine='fuzz')

        if self.sentiment_column_name is None:
            raise ValueError(
                f"Invalid `sentiment_on`. Choose from {set(self.reviews.sentiment_column_names)}.")
        else:
            if self.reviews.preprocd_data is None:
                self.data = None
            else:
                if 'dual' in self.sentiment_column_name:
                    if self.reviews.preprocd_data_ is None:
                        self.reviews.determine_sentiment(dual_scale=True)
                    self.reviews._exchange_preprocd_data_given_dual_scale()
                self.data = self.reviews.preprocd_data.copy()

        if review_column_name is None:
            self.review_column_name = self.REVIEW_COLUMN_NAME
        else:
            self.review_column_name = review_column_name

    def _print_init_values(self):
        print(f"Sentiment column name: \"{self.sentiment_column_name}\".")
        print(f"Review text column name: \"{self.review_column_name}\"")
        print(f"Random state: {self.random_state}")
        print(f"Data shape: {self.data.shape}")

    def cd_models(self, *subdir, **kwargs):
        """
        Change to the directory where the models and their relevant files are saved.

        :param subdir: name of directory or names of directories (and/or a filename)
        :type subdir: str
        :param kwargs: [optional] parameters of `src.processor._Base.cdd`
        :return: pathname of the directory for storing models
        :rtype: str
        """

        pathname = self.reviews.cdd("models", *subdir, **kwargs)

        return pathname

    def _assert_sentiment(self, sentiment):
        sentiment_ = find_similar_str(sentiment.lower(), self.VALID_SENTIMENT_LABELS, engine='fuzz')

        assert sentiment_ in self.VALID_SENTIMENT_LABELS, \
            f"`sentiment` must be one of {self.VALID_SENTIMENT_LABELS}."

        self.sentiment = sentiment
