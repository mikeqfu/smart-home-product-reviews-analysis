"""
A module that implements logistic regression models.
"""

import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

from src.modeller._base import _Base


class LogisticRegressionModel(_Base):
    """
    A class for instantiating a logistic regression model for the review texts.
    """

    #: str: Name of the model.
    NAME = 'Logistic Regression'

    def __init__(self, product_category, product_type, random_state=0, **kwargs):
        """
        :param product_category: Product category.
        :type product_category: str
        :param product_type: product type, valid values include ``{'Robotic', 'Traditional'}``
        :type product_type: str
        :param random_state: random seed number, defaults to ``0``
        :type random_state: int or None
        :param kwargs: [optional] parameters of the class :class:`~src.modeller._Base`

        :ivar sklearn.feature_extraction.text.CountVectorizer word_vectorizer:
            A collection of text documents represented as a matrix of token counts.
        :ivar scipy.sparse.csr_matrix word_counter: Document-term matrix.

        :ivar sklearn.linear_model.LogisticRegression or None logit:
            Object of logistic regression model.
        :ivar float or None score: Mean accuracy on test data.
        :ivar list or None coefficients: Estimated coefficients.
        :ivar list or None odds_ratios: Odds ratios.
        :ivar pandas.DataFrame or None summary: Summary of model coefficients.

        **Examples**::

            >>> from src.modeller import LogisticRegressionModel
            >>> logit_rvc = LogisticRegressionModel('vacuum', product_type='robotic')
            >>> logit_rvc.NAME
            'Logistic Regression'
            >>> logit_rvc.review_column_name
            'review_text'
            >>> logit_rvc.sentiment_column_name
            'sentiment_on_dual_scale'
        """

        super().__init__(product_category, product_type, random_state=random_state, **kwargs)

        self.word_vectorizer = CountVectorizer(dtype=np.uint8)
        self.word_counter = None

        self.logit = None
        self.score = None
        self.coefficients = None
        self.odds_ratios = None
        self.summary = None

    def logistic_regression(self, test_size=.15, feature_scaled=True, cv=None, solver='saga',
                            max_iter=10000, n_jobs=None, verbose=False, ret_summary=False,
                            **kwargs):
        """
        An example model: a multinomial logistic regression model.

        :param test_size: proportion of a test set, defaults to ``.15``
        :type test_size: float
        :param feature_scaled: whether to scale the feature data, defaults to ``True``
        :type feature_scaled: bool
        :param cv: ``cv`` of the class `sklearn.linear_model.LogisticRegressionCV`_, defaults to ``None``
        :type cv: int or None
        :param solver: name of solver, defaults to ``'saga'``
        :type solver: str
        :param max_iter: maximum number of iteration, defaults to ``5000``
        :type max_iter: int
        :param n_jobs: defaults to ``6``
        :type n_jobs: int or None
        :param verbose: whether to print relevant information in console, defaults to ``False``
        :type verbose: bool or int
        :param ret_summary: whether to return a summary of estimated coefficients, defaults to ``False``
        :type ret_summary: bool
        :param kwargs: [optional] parameters of `sklearn.linear_model.LogisticRegression`_

        .. _`sklearn.linear_model.LogisticRegression`:
            https://scikit-learn.org/stable/modules/generated/
            sklearn.linear_model.LogisticRegression.html
        .. _`sklearn.linear_model.LogisticRegressionCV`:
            https://scikit-learn.org/stable/modules/generated/
            sklearn.linear_model.LogisticRegressionCV.html

        **Examples**::

            >>> from src.modeller import LogisticRegressionModel
            >>> logit_rvc = LogisticRegressionModel(product_type='Robotic')
            >>> logit_rvc.logistic_regression(verbose=True)
            >>> print('Mean accuracy: %.2f%%' % (logit_rvc.score * 100))
            Mean accuracy: 95.91%
            >>> logit_rvc.summary
                  feature_name  coef_positive  coef_neutral  coef_negative
            0            great      12.758334     -1.925204     -10.833130
            1             love       9.943116     -1.847491      -8.095624
            2             easy       6.910460     -1.138867      -5.771593
            3          amazing       5.308514     -0.841767      -4.466747
            4             well       4.889721     -1.484675      -3.405046
                        ...            ...           ...            ...
            21938       return      -4.118999     -0.010412       4.129411
            21939         dead      -4.137166     -0.349761       4.486927
            21940     horrible      -4.155905     -0.349945       4.505850
            21941         stop      -4.282742      0.665926       3.616816
            21942      useless      -4.409014     -0.188151       4.597165
            [21943 rows x 4 columns]
            >>> logit_tvc = LogisticRegressionModel(product_type='Traditional')
            >>> logit_tvc.logistic_regression(verbose=True)
            >>> print('Mean accuracy: %.2f%%' % (logit_tvc.score * 100))
            Mean accuracy: 96.02%
            >>> logit_tvc.summary
                    feature_name  coef_positive  coef_neutral  coef_negative
            0               easy      11.456976     -1.481144      -9.975831
            1               love      10.424510     -2.368445      -8.056065
            2              great       8.296676     -2.157647      -6.139029
            3            amazing       6.222923     -1.157312      -5.065612
            4               well       5.916765     -1.332715      -4.584049
                          ...            ...           ...            ...
            21390  disappointing      -3.376083      0.898763       2.477320
            21391       terrible      -4.017264     -0.722647       4.739912
            21392       horrible      -4.205845     -0.143932       4.349778
            21393           poor      -4.870857     -0.408490       5.279347
            21394         return      -5.675754      0.858362       4.817391
            [21395 rows x 4 columns]
        """

        # test_size = .15
        # random_state = 0
        # max_iter = 10000
        # n_jobs = 6
        # verbose = True
        # solver = 'saga'

        if self.word_counter is None:
            self.word_counter = self.word_vectorizer.fit_transform(
                self.data[self.review_column_name].values)

        # X_train_, X_test, y_train_, y_test = train_test_split(
        #     word_counter, data[sentiment_col], test_size=test_size, random_state=random_state)
        #
        # X_train, X_val, y_train, y_val = train_test_split(
        #     X_train_, y_train_, test_size=test_size, random_state=random_state)

        X_train, X_test, y_train, y_test = train_test_split(
            self.word_counter, self.data[self.sentiment_column_name], test_size=test_size,
            random_state=self.random_state)

        if feature_scaled:
            max_abs_scaler = MaxAbsScaler()
            X_train, X_test = map(max_abs_scaler.fit_transform, [X_train, X_test])

        n_jobs_ = os.cpu_count() - 1 if n_jobs is None else n_jobs
        lr = LogisticRegression(
            solver=solver, max_iter=max_iter, random_state=self.random_state, verbose=verbose,
            n_jobs=n_jobs_, **kwargs)

        if cv:
            lr = LogisticRegressionCV(
                cv=cv, solver=solver, max_iter=max_iter, random_state=self.random_state,

                verbose=verbose, n_jobs=n_jobs_, **kwargs)

        lr.fit(X_train, y_train)

        self.score = lr.score(X_test, y_test)

        self.coefficients = lr.intercept_.tolist() + lr.coef_[0].tolist()

        self.odds_ratios = np.exp(self.coefficients).tolist()

        feature_names = self.word_vectorizer.get_feature_names_out()

        coefficients = {
            'feature_name': feature_names,
            'coef_positive': lr.coef_[2],
            'coef_neutral': lr.coef_[1],
            'coef_negative': lr.coef_[0],
        }

        summary = pd.DataFrame(coefficients)
        self.summary = summary.sort_values('coef_positive', ascending=False, ignore_index=True)

        self.logit = lr

        if ret_summary:
            return self.summary
