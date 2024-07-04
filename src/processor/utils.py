"""
A module that contains utilities for assisting in data processing.
"""

import re

import dateutil.parser
import pandas as pd
from pyhelpers._cache import _check_dependency
from pyhelpers.ops import update_dict_keys
from pyhelpers.text import numeral_english_to_arabic
from vaderSentiment import vaderSentiment

from src._cache import MOSES_DETOKENIZER, MOSES_TOKENIZER, TYPO_CORRECTIONS
from src.utils import lemmatize_text, normalise_text, remove_stopwords


def parse_review_date(review_date):
    """
    Parse a single record of review dates.

    :param review_date: date when a review was made
    :type review_date: str or None
    :return: parsed data of the given input, including both the date and location of the review
    :rtype: list

    **Examples**::

        >>> from src.processor.utils import parse_review_date
        >>> parse_review_date('December 6, 2020')
        [datetime.datetime(2020, 12, 6, 0, 0), 'United States']
        >>> parse_review_date('Reviewed in the United Kingdom on August 20, 2020')
        [datetime.datetime(2020, 8, 20, 0, 0), 'United Kingdom']
    """

    date_pattern = re.compile(
        r'(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|'
        r'Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?)'
        r'\s+\d{1,2},\s+\d{4}')
    location_pattern = re.compile(r'Reviewed in( the)? (.*?) on ')

    if re.match(r"Reviewed in .* on " + date_pattern.pattern, review_date):
        date = re.search(date_pattern, review_date).group(0)
        review_location = re.search(location_pattern, review_date.replace(date, '')).group(2)
        date_and_location = [dateutil.parser.parse(date), normalise_text(review_location)]

    elif re.match(date_pattern, review_date):
        date_and_location = [dateutil.parser.parse(review_date), 'United States']

    else:
        print(f"Check \"{review_date}\"")
        date_and_location = [None, '']

    return date_and_location


def regulate_people_found_helpful(people_found_helpful):
    """
    Regulate a single record of the number of people who found a piece of review was helpful.

    :param people_found_helpful: raw data of the number of people who found a review was helpful
    :type people_found_helpful: str
    :return: parsed data of the given input
    :rtype: int

    **Examples**::

        >>> from src.processor.utils import regulate_people_found_helpful
        >>> regulate_people_found_helpful('One')
        1
        >>> regulate_people_found_helpful('')
        0
    """

    if pd.isna(people_found_helpful):
        people_found_helpful_ = 0

    else:
        try:
            people_found_helpful_ = int(people_found_helpful.replace(',', ''))
        except ValueError:
            people_found_helpful_ = numeral_english_to_arabic(people_found_helpful)

    return people_found_helpful_


def correct_identified_typos(review_text, split=True):
    # noinspection SpellCheckingInspection
    """
    Correct typos that have been identified.

    :param review_text: textual data of product reviews
    :type review_text: str
    :param split: whether to split the corrected review text into a list of words, defaults to ``True``
    :type split: bool
    :return: textual data of which identified typos are corrected
    :rtype: list or str

    **Examples**::

        >>> from src.processor.utils import correct_identified_typos
        >>> original_review_text = 'I would like to replce it.'
        >>> correct_identified_typos(original_review_text)
        ['I', 'would', 'like', 'to', 'replace', 'it', '.']
        >>> correct_identified_typos(original_review_text, split=False)
        'I would like to replace it.'
    """

    review_text_ = MOSES_TOKENIZER.tokenize(review_text)

    for i, x in enumerate(review_text_):
        for k in TYPO_CORRECTIONS.keys():
            # if x in {k, k + ",", k + "."}:  # or re.match(r'%s[ %s]' % (k, string.punctuation), x):
            if k == x:
                review_text_[i] = TYPO_CORRECTIONS[k]

    if not split:
        review_text_ = MOSES_DETOKENIZER.detokenize(review_text_)

    return review_text_


def get_vader_sentiment_score(review_text, use_nltk=False, prefix='vs', suffix='score'):
    """
    Compute sentiment scores for product reviews using
    `vaderSentiment <https://pypi.org/project/vaderSentiment/>`_ or
    `nltk.sentiment.vader <https://www.nltk.org/_modules/nltk/sentiment/vader.html>`_.

    :param review_text: textual data of product reviews
    :type review_text: str or pandas.Series or typing.List[str] or typing.Tuple[str]
    :param use_nltk: whether to use the `vader`_ module of `NLTK`_, defaults to ``False``
    :type use_nltk: bool
    :param prefix: prefix to a column/key name for a VADER sentiment score, defaults to ``'vs'``
    :type prefix: str
    :param suffix: suffix to a column/key name for a VADER sentiment score, defaults to ``'score'``
    :type suffix: str
    :return: VADER sentiment scores
    :rtype: pandas.DataFrame

    .. _`vader`: https://www.nltk.org/_modules/nltk/sentiment/vader.html
    .. _`NLTK`: https://www.nltk.org/

    **Examples**::

        >>> from src.processor.utils import get_vader_sentiment_score
        >>> from src.utils import remove_stopwords, lemmatize_text
        >>> from pyhelpers.store import load_pickle
        >>> # Get a random example of the reviews
        >>> example_review_text = load_pickle("src\\data\\example_review_text.pkl")
        >>> example_review_text
        136576    I wish I would spent a little more money and b...
        Name: ReviewText, dtype: object
        >>> x1 = example_review_text.values[0]
        >>> x1
        "I wish I would spent a little more money and bought one with higher power. We have 2 Gol...
        >>> get_vader_sentiment_score(x1)
        {'vs_neg_score': 0.0,
         'vs_neu_score': 0.87,
         'vs_pos_score': 0.13,
         'vs_compound_score': 0.4019}
        >>> x2 = lemmatize_text(x1)
        >>> x2
        'wish spend little more money buy high power have'
        >>> get_vader_sentiment_score(x2)
        {'vs_neg_score': 0.0,
         'vs_neu_score': 0.748,
         'vs_pos_score': 0.252,
         'vs_compound_score': 0.4019}
        >>> x3 = remove_stopwords(x1)
        >>> x3
        "wish would spent little money bought one higher power. 2 Golden's."
        >>> get_vader_sentiment_score(x3)
        {'vs_neg_score': 0.0,
         'vs_neu_score': 0.787,
         'vs_pos_score': 0.213,
         'vs_compound_score': 0.4019}
        >>> get_vader_sentiment_score(example_review_text)
                vs_neg_score  vs_neu_score  vs_pos_score  vs_compound_score
        136576           0.0          0.87          0.13             0.4019
        >>> get_vader_sentiment_score(example_review_text.map(remove_stopwords))
                vs_neg_score  vs_neu_score  vs_pos_score  vs_compound_score
        136576           0.0         0.787         0.213             0.4019

    .. note::

        For the use of `vaderSentiment <https://pypi.org/project/vaderSentiment/>`_,
        we shall input the original sentence, without removing any 'unnecessary' words
        (e.g. stopwords) or punctuation.
    """

    if use_nltk:
        nltk_sentiment = _check_dependency(name='nltk.sentiment')
        vader = nltk_sentiment.vader.SentimentIntensityAnalyzer()
    else:
        vader = vaderSentiment.SentimentIntensityAnalyzer()

    if isinstance(review_text, str):
        vs_scores_ = vader.polarity_scores(review_text)

        repl = {k: '_'.join([prefix, k, suffix]) for k in vs_scores_.keys()}
        vs_scores = update_dict_keys(vs_scores_, replacements=repl)

    else:
        if isinstance(review_text, pd.Series):
            vs_scores_ = review_text.map(vader.polarity_scores)
            vs_scores = pd.DataFrame(vs_scores_.tolist(), index=vs_scores_.index)
        else:
            vs_scores_ = [vader.polarity_scores(x) for x in review_text]
            vs_scores = pd.DataFrame(vs_scores_)

        vs_scores.columns = ['_'.join([prefix, x, suffix]) for x in vs_scores.columns]

    return vs_scores


def determine_sentiment_on_rating(rating, pos=4, neg=2):
    """
    Get a sentiment indicator according to product rating.

    :param rating: rating associated with a product review, including 1, 2, 3, 4 and 5
    :type rating: int or float
    :param pos: threshold value (between 3 and 5) for deciding on positive reviews,
        defaults to ``4.``; when ``rating`` is higher than or equal to ``pos``
    :type pos: int or float
    :param neg: threshold value (between 1 and 3) for deciding on negative reviews,
        defaults to ``2.``; when ``rating`` is lower than or equal to ``neg``
    :type neg: int or float
    :return: a sentiment score based on rating data
    :rtype: int

    **Examples**::

        >>> from src.processor.utils import determine_sentiment_on_rating
        >>> determine_sentiment_on_rating(rating=5)
        'positive'
        >>> determine_sentiment_on_rating(rating=4)
        'positive'
        >>> determine_sentiment_on_rating(rating=3)
        'neutral'
        >>> determine_sentiment_on_rating(rating=2)
        'negative'
        >>> determine_sentiment_on_rating(rating=1)
        'negative'
    """

    if rating >= pos:
        sentiment = 'positive'
    elif rating <= neg:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    return sentiment


def determine_sentiment_on_vs_score(vs_score, pos=0.05, neg=-0.05):
    """
    Get a sentiment indicator according to a sentiment score which is computed by
    `vaderSentiment`_ or `nltk.sentiment.vader`_.

    :param vs_score: compound score (computed by `vaderSentiment`_ or `nltk.sentiment.vader`_;
        between -1 and 1) associated with a product review
    :type vs_score: float
    :param pos: threshold value (between -1 and 1) for deciding on positive reviews,
        defaults to ``0.05``; when ``vs_score`` is higher than or equal to ``pos``
    :type pos: int or float
    :param neg: threshold value (between -1 and 1) for deciding on negative reviews,
        defaults to ``-0.05``; when ``vs_score`` is lower than or equal to ``neg``
    :type neg: int or float
    :return: a sentiment based on scores,
        which are computed by `vaderSentiment`_ or `nltk.sentiment.vader`_
    :rtype: int

    .. _`vaderSentiment`: https://pypi.org/project/vaderSentiment/
    .. _`nltk.sentiment.vader`: https://www.nltk.org/_modules/nltk/sentiment/vader.html

    **Examples**::

        >>> from src.processor.utils import determine_sentiment_on_vs_score, get_vader_sentiment_score
        >>> from pyhelpers.store import load_pickle
        >>> example_review_text = load_pickle("src\\data\\example_review_text.pkl")
        >>> example_review_text
        136576    I wish I would spent a little more money and b...
        Name: ReviewText, dtype: object
        >>> example_vs_scores = get_vader_sentiment_score(example_review_text)
        >>> example_vs_scores
                vs_neg_score  vs_neu_score  vs_pos_score  vs_compound_score
        136576           0.0          0.87          0.13             0.4019
        >>> example_vs_scores['vs_compound_score'].map(determine_sentiment_on_vs_score)
        136576    positive
        Name: vs_compound_score, dtype: object
    """

    if vs_score >= pos:
        sentiment = 'positive'
    elif vs_score <= neg:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    return sentiment
